"""BM25 search over policy chunks, plus lookups of named policies and NPPF paragraphs."""

import math
import re
from collections import Counter
from collections.abc import Sequence

from config import Settings
from domain import LOCAL_PLAN, NPPF, PolicyChunk
from llm import LLMClient
from policy.chunking import load_chunks
from policy.config import PolicyConfig
from policy.embeddings import EmbeddingIndex, load_embedding_index
from policy.figures import caption_figures
from policy.fuzzy import FuzzyIndex
from policy.ocr import transcribe_scanned_sources
from policy.tables import refine_tables

FUZZY_DEPTH = 200  # fuzzy matches considered per query
FUZZY_EXTRAS = 0.5  # extras the fuzzy ranking adds, as a fraction of k
EMBED_EXTRAS = 0.5  # extras the embedding ranking adds, as a fraction of k

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "in",
        "a",
        "for",
        "is",
        "on",
        "be",
        "that",
        "with",
        "as",
        "by",
        "are",
        "this",
        "or",
        "an",
        "at",
        "it",
        "from",
        "will",
        "should",
        "which",
        "not",
        "must",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class PolicyIndex:
    """BM25 ranking over policy chunks."""

    K1, B = 1.5, 0.75

    def __init__(
        self, chunks: list[PolicyChunk], opt_in_labels: frozenset[str] = frozenset()
    ) -> None:
        self.chunks = chunks
        self._opt_in = opt_in_labels
        self._fuzzy: FuzzyIndex | None = None  # built on first hybrid search
        self.embeddings: EmbeddingIndex | None = None  # set by `with_embeddings`
        self._units: dict[str, list[PolicyChunk]] = {}
        for chunk in chunks:
            if chunk.unit_id:
                self._units.setdefault(chunk.unit_id, []).append(chunk)
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    async def aload(
        cls, settings: Settings, client: LLMClient, config: PolicyConfig
    ) -> "PolicyIndex":
        """Transcribe scanned sources (cached), then load the index.

        With `settings.refine_tables`, flattened tables are first rebuilt as markdown
        by the cheap model (see `policy.tables`). With `settings.caption_figures`, figures
        in `captions: true` sources are described by a vision model (`policy.figures`).
        Replies are cached, so both are one-off costs.
        """
        await transcribe_scanned_sources(settings, client, config.sources)
        chunks = load_chunks(settings, config.sources)
        if settings.refine_tables:
            chunks = await refine_tables(
                client, settings.table_model, chunks, settings.caption_reasoning_off
            )
        if settings.caption_figures:
            chunks = [*chunks, *await caption_figures(settings, client, config.sources)]
        return cls(chunks, config.opt_in_labels).with_embeddings(settings)

    @classmethod
    def load(cls, settings: Settings, config: PolicyConfig) -> "PolicyIndex":
        """Load the index for `config`'s sources (chunks are cached on disk)."""
        return cls(load_chunks(settings, config.sources), config.opt_in_labels)

    def with_embeddings(self, settings: Settings) -> "PolicyIndex":
        """Attach the local embedding ranking if `settings.embeddings` (else unchanged)."""
        if settings.embeddings:
            self.embeddings = load_embedding_index(
                self.chunks, settings.embedding_model, settings.cache_dir
            )
        return self

    def _allowed(self, sources: set[str] | None) -> list[int]:
        """Chunk positions a search may return.

        Without `sources`, large opt-in documents (legislation) are skipped so they do
        not drown out policy; name them explicitly to search them.
        """
        out = []
        for i, chunk in enumerate(self.chunks):
            if sources:
                if chunk.source in sources:
                    out.append(i)
            elif chunk.source not in self._opt_in:
                out.append(i)
        return out

    def _bm25_rank(self, query: str, allowed: list[int]) -> list[int]:
        """Allowed chunk positions with a BM25 score above zero, best first."""
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i in allowed:
            tf = self._tf[i]
            score = 0.0
            norm = self.K1 * (1 - self.B + self.B * self._len[i] / self._avg)
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self._idf[t] * f * (self.K1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    def _fuzzy_rank(self, query: str, allowed: list[int]) -> list[int]:
        """Allowed chunk positions ranked by the stem/trigram index, best first."""
        if self._fuzzy is None:
            self._fuzzy = FuzzyIndex([c.text for c in self.chunks], tokenize)
        scores = self._fuzzy.scores(query)
        keep = set(allowed)
        ranked = sorted(((s, i) for i, s in scores.items() if i in keep), reverse=True)
        return [i for _, i in ranked[:FUZZY_DEPTH]]

    def search(
        self, query: str, k: int = 5, sources: set[str] | None = None
    ) -> list[PolicyChunk]:
        """Top-k chunks for a query by BM25 alone.

        Without `sources`, large opt-in documents (legislation) are skipped so they
        do not drown out policy; name them explicitly to search them.
        """
        ranked = self._bm25_rank(query, self._allowed(sources))
        return [self.chunks[i] for i in ranked[:k]]

    def hybrid_search(
        self, queries: Sequence[str], k: int = 5, sources: set[str] | None = None
    ) -> list[PolicyChunk]:
        """BM25's top-k for each query, plus extras the fuzzy and embedding rankings find.

        The rankers complement each other rather than compete: BM25 is the more precise
        and keeps its place, while the stem/trigram index adds `FUZZY_EXTRAS * k` chunks
        per query that BM25 missed (other word forms, split or mistyped words). Fusing
        the two rankings by rank instead lowered recall on the eval cases, because the
        fuzzy ranking is the noisier of the two. The local embedding ranking (when
        attached) is a third source of extras after the fuzzy ones: it finds chunks
        that share meaning but no words with the query.
        """
        allowed = self._allowed(sources)
        extras = max(1, round(k * FUZZY_EXTRAS))
        chosen: list[int] = []
        seen: set[int] = set()
        for query in queries:
            for i in self._bm25_rank(query, allowed)[:k]:
                if i not in seen:
                    seen.add(i)
                    chosen.append(i)
        for query in queries:
            added = 0
            for i in self._fuzzy_rank(query, allowed):
                if added >= extras:
                    break
                if i not in seen:
                    seen.add(i)
                    chosen.append(i)
                    added += 1
        if self.embeddings is not None:
            for query in queries:
                added = 0
                for i in self.embeddings.rank(query, allowed):
                    if added >= extras:
                        break
                    if i not in seen:
                        seen.add(i)
                        chosen.append(i)
                        added += 1
        return [self.chunks[i] for i in chosen]

    def policy_text(
        self, source: str, policy_prefix: str, limit: int = 3500
    ) -> list[PolicyChunk]:
        """All chunks whose policy id starts with `policy_prefix` (for curated lookups).

        The policy's own clauses are always returned in full, however long: cutting a
        policy mid-way would hide conditions the assessor must apply. `limit` bounds
        only what is added around them (supporting text and tables), in page order.
        """
        hits = [
            c
            for c in self.chunks
            if c.source == source
            and c.policy_id
            and c.policy_id.startswith(policy_prefix)
        ]
        out = [c for c in hits if c.kind == "policy"]
        size = sum(len(c.text) for c in out)
        for chunk in hits:
            if chunk.kind == "policy":
                continue
            if size + len(chunk.text) > limit:
                break
            out.append(chunk)
            size += len(chunk.text)
        return out

    def unit_parts(self, chunk: PolicyChunk) -> list[PolicyChunk]:
        """The other parts of the unit `chunk` belongs to (a split policy or table)."""
        if not chunk.unit_id:
            return []
        return sorted(
            (c for c in self._units[chunk.unit_id] if c is not chunk),
            key=lambda c: c.part,
        )

    def nppf_paragraph(self, number: int, limit: int = 1800) -> list[PolicyChunk]:
        """NPPF chunks whose detected paragraph range covers `number`."""
        out, size = [], 0
        for chunk in self.chunks:
            if chunk.source != NPPF or not chunk.policy_id:
                continue
            nums = [int(n) for n in re.findall(r"\d+", chunk.policy_id)]
            if nums and nums[0] <= number <= nums[-1]:
                if size + len(chunk.text) > limit:
                    break
                out.append(chunk)
                size += len(chunk.text)
        return out

    def local_plan_heading(self, number: int) -> str | None:
        """Full heading of Local Plan policy `number`, e.g. 'Policy 57: Flood Risk ...'."""
        prefix = f"Policy {number}:"
        for chunk in self.chunks:
            if (
                chunk.source == LOCAL_PLAN
                and chunk.policy_id
                and chunk.policy_id.startswith(prefix)
            ):
                return str(chunk.policy_id)
        return None

    def nppf_paragraph_text(
        self, number: int, limit: int = 700
    ) -> tuple[int, str] | None:
        """(page, text) of NPPF paragraph `number`, or None if it cannot be found.

        Skips contents entries and chapter headings, which share numbers with
        paragraphs (chapter 11 vs paragraph 11): a real paragraph opens with a
        sentence of at least five words that is not a bare page reference.
        """
        marker = re.compile(rf"(?m)^[ \t]*{number}\.[ \t]+(?P<first>[^\n]*)")
        following = re.compile(rf"(?m)^[ \t]*{number + 1}\.[ \t]+")
        nppf = [c for c in self.chunks if c.source == NPPF]
        for i, chunk in enumerate(nppf):
            for match in marker.finditer(chunk.text):
                if not _is_paragraph_opening(match.group("first")):
                    continue
                text = chunk.text[match.start() :]
                j = i + 1
                while (
                    len(text) < limit
                    and j < len(nppf)
                    and nppf[j].page - chunk.page <= 1
                ):
                    text += "\n" + nppf[j].text
                    j += 1
                cut = following.search(text)
                if cut:
                    text = text[: cut.start()]
                return chunk.page, " ".join(text.split())[:limit]
        return None


def _is_paragraph_opening(first_line: str) -> bool:
    """True for the first line of a real NPPF paragraph (not a contents/heading line)."""
    line = first_line.strip()
    if re.search(r"\s\d{1,3}$", line):  # contents entry ending in a page number
        return False
    words = len(line.split())
    return words >= 8 or (words >= 5 and line.endswith((":", ".", ";", ",")))
