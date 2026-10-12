"""Turn the parsed policy PDFs into structure-aware chunks that carry citation metadata.

The parser gives one text blob per page, so structure is recovered from the text:

1. Pages become one stream of lines, so a policy or table that runs over a page break
   stays together. Running headers, bare page numbers and extractor artefacts go.
2. The stream is cut into *units* at detected boundaries: a policy's clauses, its
   supporting text, a table (caption to the next heading or numbered paragraph), or
   plain text. The Local Plan and the NPPF have their own segmenters because their
   numbering is how officers cite them.
3. A unit is packed into chunks of at most `CHUNK_CHARS`, splitting only at clause or
   paragraph boundaries (never inside a table row or mid-sentence if avoidable). Every
   continuation repeats the unit's heading and records `part i/n`, so any chunk can be
   read alone and retrieval can fetch its siblings through `unit_id`.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from config import Settings
from domain import LOCAL_PLAN, NPPF, ChunkKind, PolicyChunk
from policy.config import PolicySource
from policy.ocr import load_ocr_pages

CHUNK_CHARS = 1800
TABLE_CHARS = 6000  # a table is kept whole up to this size
OVERLAP_CHARS = 150  # context carried into continuation parts of plain text
MAX_TABLE_LINES = 600  # safety stop if a table's end is never detected
MAX_TABLE_PAGES = (
    2  # a table is not expected to run over more than this many page breaks
)
PROSE_LINE = 75  # characters; table cells wrap shorter than running prose does
PROSE_RUN = 3  # long lines among the last few that mean the table has turned to prose

_RUNNING_HEADER = re.compile(r"^\d{1,3}\s*\[go to Contents\]$", re.I)
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_LEADERS = re.compile(r"\.{4,}|(?:\.\s){4,}")
_CONTENTS_ENTRY = re.compile(r"^\d{1,2}\.\s+[A-Z][^.]{3,70}\s\d{1,3}$")
_CHAPTER = re.compile(r"^Chapter\s+\d+\s*[:.\-]", re.I)
_POLICY = re.compile(r"^Policy\s+(\d{1,3})\s*:\s*([A-Z].{2,})$")
_EXPLANATION = re.compile(
    r"^(?:Explanation|Supporting Text|Justification|Reasoned Justification)$", re.I
)
_NUMBERED_PARA = re.compile(r"^\d{1,3}\.\d{1,3}\.?\s+\S")
_CLAUSE = re.compile(r"^(?:[A-Za-z]\)|\([A-Za-z0-9]{1,4}\)|\d{1,2}\)|-\s)\s*\S")
_TABLE_CAPTION = re.compile(r"^Table\s+\d+(?:\.\d+)?\s*[:.]\s*\S")
_NPPF_HEADING = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Za-z0-9 ,'’&/\-]{3,70})$")
_NPPF_PARA = re.compile(r"^(\d{1,3})\.\s+\S")
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")

_ARTEFACTS = {
    "\r\n": "\n",
    "\r": "\n",
    "\t": " ",
    "\x02": "-",  # soft hyphen the extractor leaves inside words ("off\x02site")
    "­": "",
    "": "- ",  # private-use bullets
    "": "- ",
}


@dataclass(frozen=True)
class _Line:
    text: str
    page: int


@dataclass
class _Unit:
    """A run of lines that belongs together: one policy, one table, one paragraph."""

    kind: ChunkKind
    policy_id: str | None = None
    heading: str | None = None
    lines: list[_Line] = field(default_factory=list)


@dataclass(frozen=True)
class _Piece:
    """The smallest thing the packer will not split: a clause or paragraph."""

    text: str
    first_page: int
    last_page: int


# --------------------------------------------------------------------- page text


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _page_text(page: dict[str, object]) -> str:
    """Join a page's text blocks, dropping the extractor's duplicate renderings.

    Besides exact repeats, a block typed `table` is dropped when nearly all of its words
    already appear in the page's other blocks: the extractor often emits ordinary prose
    twice, once as a scrambled "table".
    """
    blocks = [
        b
        for b in page["blocks"]  # type: ignore[attr-defined]
        if b.get("block_type") == "text" and b["text"].strip()
    ]
    seen: set[str] = set()
    kept = []
    for block in blocks:
        key = " ".join(block["text"].split())
        if key in seen:
            continue
        if block.get("source_type") == "table":
            others: set[str] = set()
            for other in blocks:
                if other is not block:
                    others |= _tokens(other["text"])
            mine = _tokens(block["text"])
            if mine and len(mine & others) / len(mine) >= 0.9:
                continue
        seen.add(
            key
        )  # only kept blocks count, so a dropped copy cannot hide the original
        kept.append(block["text"])
    return "\n".join(kept)


def _clean(text: str) -> str:
    for bad, good in _ARTEFACTS.items():
        text = text.replace(bad, good)
    return text


def _page_lines(text: str, page: int) -> list[_Line]:
    """Lines of one page without running headers or the bare page number."""
    lines = []
    first = True
    for raw in _clean(text).split("\n"):
        line = " ".join(raw.split())
        if _RUNNING_HEADER.match(line):
            continue
        if first and line:
            first = False
            if _PAGE_NUMBER.match(line):
                continue
        lines.append(_Line(line, page))
    return lines


def _contents_pages(lines: list[_Line]) -> set[int]:
    """Pages dominated by dotted-leader lines: tables of contents, not policy."""
    counts: dict[int, int] = {}
    entries: dict[int, int] = {}
    for line in lines:
        if _LEADERS.search(line.text):
            counts[line.page] = counts.get(line.page, 0) + 1
        elif _CONTENTS_ENTRY.match(line.text):  # "1. Introduction 5", no leaders
            entries[line.page] = entries.get(line.page, 0) + 1
    return {p for p, n in counts.items() if n >= 3} | {
        p for p, n in entries.items() if n >= 5
    }


# ------------------------------------------------------------------ segmentation


def _table_ends(table: _Unit, line: _Line) -> bool:
    """True if `line` is past the end of the open table.

    Tables have no closing marker, so the end is inferred: several long lines close
    together (running prose, since wrapped cells are short and only occasionally hold a
    long line), a table that has outrun the pages one could span, or a size safety
    stop. Headings and numbered paragraphs are checked by the callers.
    """
    recent = [*table.lines[-5:], line]
    long_lines = sum(1 for ln in recent if len(ln.text) >= PROSE_LINE)
    return (
        len(table.lines) > MAX_TABLE_LINES
        or line.page > table.lines[0].page + MAX_TABLE_PAGES
        or (len(table.lines) > 1 and long_lines >= PROSE_RUN)
    )


def _take_prose_tail(table: _Unit) -> list[_Line]:
    """Remove the long lines at the end of a closed table; they belong to what follows."""
    tail: list[_Line] = []
    while len(table.lines) > 1 and len(table.lines[-1].text) >= PROSE_LINE:
        tail.insert(0, table.lines.pop())
    return tail


def _segment_local_plan(lines: list[_Line]) -> list[_Unit]:
    """Cut the Local Plan into policy, supporting-text, table and plain-text units."""
    units: list[_Unit] = []
    contents = _contents_pages(lines)
    cur: _Unit | None = None
    policy: str | None = None
    heading: str | None = None
    before_table: ChunkKind = "text"
    seen_policies: set[int] = (
        set()
    )  # a heading is accepted once; repeats are table rows

    def new_policy(text: str) -> re.Match[str] | None:
        """A policy heading, if this policy number has not been seen before."""
        match = _POLICY.match(text)
        return match if match and int(match.group(1)) not in seen_policies else None

    def open_unit(kind: ChunkKind, head: str | None = None) -> _Unit:
        unit = _Unit(kind, policy, head if head is not None else heading)
        units.append(unit)
        return unit

    for line in lines:
        text = line.text
        if line.page in contents or not text:
            if cur is None:
                cur = open_unit("text")
            cur.lines.append(line)
            continue
        if cur is not None and cur.kind == "table":
            structural = (
                _NUMBERED_PARA.match(text)
                or new_policy(text)
                or _CHAPTER.match(text)
                or _EXPLANATION.match(text)
                or _TABLE_CAPTION.match(text)
            )
            inferred = not structural and _table_ends(cur, line)
            if structural or inferred:
                tail = _take_prose_tail(cur) if inferred else []
                cur = open_unit(before_table)
                cur.lines.extend(tail)
        if _CHAPTER.match(text):
            policy, heading = None, text
            cur = open_unit("text", text)
        elif match := new_policy(text):
            seen_policies.add(int(match.group(1)))
            policy = f"Policy {match.group(1)}: {match.group(2).strip()[:90]}"
            heading = policy
            cur = open_unit("policy")
        elif _TABLE_CAPTION.match(text):
            before_table = cur.kind if cur is not None else "text"
            if before_table == "table":
                before_table = "text"
            cur = open_unit("table", text)
        elif (
            cur is not None
            and cur.kind == "policy"
            and (_EXPLANATION.match(text) or _NUMBERED_PARA.match(text))
        ):
            cur = open_unit("supporting", "Supporting text")
        elif cur is None:
            cur = open_unit("text")
        cur.lines.append(line)
    return [u for u in units if any(ln.text for ln in u.lines)]


def _para_follows(number: str, last: int) -> bool:
    """True if `number` can be the NPPF paragraph after `last` (or the first one)."""
    value = int(number)
    return last < value <= last + 8 or (last == 0 and value <= 8)


def _segment_nppf(lines: list[_Line]) -> list[_Unit]:
    """One unit per numbered NPPF paragraph, keeping its (a), (b) sub-items.

    A paragraph number is accepted only if it follows the previous one closely, which
    rejects footnote numbers and list items that also start with a number.
    """
    units: list[_Unit] = []
    contents = _contents_pages(lines)
    heading: str | None = None
    last = 0
    cur: _Unit | None = None
    for line in lines:
        text = line.text
        if line.page in contents or not text:
            if cur is not None:
                cur.lines.append(line)
            continue
        section = _NPPF_HEADING.match(text)
        para = _NPPF_PARA.match(text)
        if section and not _para_follows(section.group(1), last):
            heading = text
            continue
        if para and _para_follows(para.group(1), last):
            last = int(para.group(1))
            cur = _Unit("policy", f"paragraph {last}", heading)
            units.append(cur)
        elif cur is None:
            cur = _Unit("text", None, heading)
            units.append(cur)
        cur.lines.append(line)
    return units


def _segment_generic(lines: list[_Line]) -> list[_Unit]:
    """Plain text split only at table captions, so tables stay whole."""
    units: list[_Unit] = []
    contents = _contents_pages(lines)
    cur: _Unit | None = None
    for line in lines:
        text = line.text
        in_contents = line.page in contents
        starts_table = bool(_TABLE_CAPTION.match(text)) and not in_contents
        inferred_end = (
            cur is not None
            and cur.kind == "table"
            and not starts_table
            and _table_ends(cur, line)
        )
        if starts_table:
            cur = _Unit("table", None, text)
            units.append(cur)
        elif cur is None or inferred_end:
            tail = _take_prose_tail(cur) if cur is not None else []
            cur = _Unit("text", lines=tail)
            units.append(cur)
        cur.lines.append(line)
    return units


# ----------------------------------------------------------------------- packing


def _pieces(unit: _Unit) -> list[_Piece]:
    """Split a unit at clause starts, numbered paragraphs and blank lines."""
    per_line = unit.kind == "table"  # raw table cells have no clause structure
    pieces: list[_Piece] = []
    buf: list[_Line] = []

    def flush() -> None:
        if buf:
            pieces.append(
                _Piece("\n".join(b.text for b in buf), buf[0].page, buf[-1].page)
            )
            buf.clear()

    for line in unit.lines:
        if not line.text:
            flush()
            continue
        starts = per_line or _CLAUSE.match(line.text) or _NUMBERED_PARA.match(line.text)
        if starts and buf:
            flush()
        buf.append(line)
    flush()
    return pieces


def _wrap(text: str, limit: int) -> list[str]:
    """Word-wrap `text` into fragments of at most `limit` characters."""
    fragments: list[str] = []
    buf = ""
    for word in text.split():
        if buf and len(buf) + 1 + len(word) > limit:
            fragments.append(buf)
            buf = ""
        buf = f"{buf} {word}".strip()
    if buf:
        fragments.append(buf)
    return fragments


def _split_long(piece: _Piece, limit: int) -> list[_Piece]:
    """Break an oversized piece at sentence ends, then at word boundaries."""
    parts: list[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(" ".join(piece.text.split())):
        for fragment in _wrap(sentence, limit):
            if buf and len(buf) + 1 + len(fragment) > limit:
                parts.append(buf)
                buf = ""
            buf = f"{buf} {fragment}".strip()
    if buf:
        parts.append(buf)
    return [_Piece(p, piece.first_page, piece.last_page) for p in parts]


def _pack(pieces: list[_Piece], limit: int) -> list[list[_Piece]]:
    """Greedily fill parts up to `limit`, never splitting a piece unless it is too big."""
    flat: list[_Piece] = []
    for piece in pieces:
        flat += _split_long(piece, limit) if len(piece.text) > limit else [piece]
    parts: list[list[_Piece]] = []
    cur: list[_Piece] = []
    size = 0
    for piece in flat:
        if cur and size + len(piece.text) + 1 > limit:
            parts.append(cur)
            cur, size = [], 0
        cur.append(piece)
        size += len(piece.text) + 1
    if cur:
        parts.append(cur)
    return parts


def _continuation_header(unit: _Unit) -> str:
    title = unit.heading if unit.kind == "table" else (unit.policy_id or unit.heading)
    return f"{title} (continued)" if title else "(continued)"


def _unit_id(source: PolicySource, page: int, seq: int, unit: _Unit) -> str:
    key = f"{source.label}|{page}|{seq}|{unit.kind}|{unit.heading}|{unit.policy_id}"
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def _chunks_for_unit(source: PolicySource, unit: _Unit, seq: int) -> list[PolicyChunk]:
    pieces = _pieces(unit)
    if not pieces:
        return []
    limit = TABLE_CHARS if unit.kind == "table" else CHUNK_CHARS
    parts = _pack(pieces, limit)
    uid = _unit_id(source, pieces[0].first_page, seq, unit)
    chunks: list[PolicyChunk] = []
    previous = ""
    for number, part in enumerate(parts, 1):
        body = "\n".join(p.text for p in part)
        if number > 1:
            lead = ""
            if unit.kind == "text" and previous:
                tail = previous[-OVERLAP_CHARS:]
                lead = "…" + tail[tail.find(" ") + 1 :] + "\n"
            body = f"{_continuation_header(unit)}\n{lead}{body}"
        elif unit.kind == "supporting" and unit.policy_id:
            body = f"{unit.policy_id} (supporting text)\n{body}"
        previous = body
        first, last = part[0].first_page, part[-1].last_page
        chunks.append(
            PolicyChunk(
                source=source.label,
                page=first,
                end_page=last if last != first else None,
                policy_id=unit.policy_id,
                text=body.strip(),
                version=source.version,
                kind=unit.kind,
                heading=unit.heading,
                unit_id=uid,
                part=number,
                parts=len(parts),
            )
        )
    return chunks


def _nppf_chunks(source: PolicySource, units: list[_Unit]) -> list[PolicyChunk]:
    """Group consecutive NPPF paragraphs into chunks labelled with their paragraph range.

    A paragraph that fits is never split across chunks. One that does not is split at
    its (a), (b) sub-items and every part keeps the paragraph's number.
    """
    chunks: list[PolicyChunk] = []
    group: list[_Unit] = []
    size = 0

    def flush() -> None:
        nonlocal group, size
        if not group:
            return
        numbers = [
            int(u.policy_id.split()[-1]) for u in group if u.policy_id is not None
        ]
        label = (
            None
            if not numbers
            else (
                f"paragraph {numbers[0]}"
                if len(set(numbers)) == 1
                else f"paragraphs {min(numbers)}-{max(numbers)}"
            )
        )
        lines = [ln for u in group for ln in u.lines]
        merged = _Unit("policy", label, group[0].heading, lines)
        pieces = _pieces(merged)
        if pieces:
            uid = _unit_id(source, pieces[0].first_page, len(chunks), merged)
            body = "\n".join(p.text for p in pieces)
            first, last = pieces[0].first_page, pieces[-1].last_page
            chunks.append(
                PolicyChunk(
                    source=source.label,
                    page=first,
                    end_page=last if last != first else None,
                    policy_id=label,
                    text=body.strip(),
                    version=source.version,
                    kind="policy",
                    heading=merged.heading,
                    unit_id=uid,
                )
            )
        group, size = [], 0

    for unit in units:
        length = sum(len(ln.text) + 1 for ln in unit.lines)
        if length > CHUNK_CHARS:
            flush()
            chunks += _chunks_for_unit(source, unit, len(chunks))
            continue
        if group and (size + length > CHUNK_CHARS or unit.heading != group[0].heading):
            flush()
        group.append(unit)
        size += length
    flush()
    return chunks


# ------------------------------------------------------------------------ public


def chunks_for_source(
    source: PolicySource,
    doc: dict[str, list[dict[str, object]]],
    ocr_pages: dict[int, str],
) -> list[PolicyChunk]:
    """Chunk one parsed document by structure rather than by page.

    Pages with no extracted text fall back to OCR transcripts when provided.
    """
    lines: list[_Line] = []
    for page in doc["pages"]:
        number = int(page["page_number"])  # type: ignore[call-overload]
        text = _page_text(page) or ocr_pages.get(number, "")  # type: ignore[arg-type]
        lines += _page_lines(text, number)
    if source.label == NPPF:
        return _nppf_chunks(source, _segment_nppf(lines))
    units = (
        _segment_local_plan(lines)
        if source.label == LOCAL_PLAN
        else _segment_generic(lines)
    )
    chunks: list[PolicyChunk] = []
    for seq, unit in enumerate(units):
        chunks += _chunks_for_unit(source, unit, seq)
    return chunks


def build_chunks(
    settings: Settings, sources: tuple[PolicySource, ...]
) -> list[PolicyChunk]:
    """Chunk every registered policy document."""
    chunks: list[PolicyChunk] = []
    for source in sources:
        path = settings.extracted_dir / source.path
        if not path.exists():
            logger.warning(
                f"Policy source missing (run `uv run parser`): {source.path}"
            )
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        chunks += chunks_for_source(source, doc, load_ocr_pages(settings, source))
    # Contents pages (dotted leaders) would otherwise masquerade as policy text.
    return [c for c in chunks if len(c.text) > 80 and c.text.count("....") < 3]


def _cache_key(settings: Settings, sources: tuple[PolicySource, ...]) -> str:
    """Changes whenever the source list, any OCR transcript or the chunker changes."""
    digest = hashlib.sha256(Path(__file__).read_bytes())  # new rules, new cache
    for source in sources:
        digest.update(
            f"{source.path}|{source.label}|{source.default_search}|{source.version}".encode()
        )
        digest.update(
            json.dumps(load_ocr_pages(settings, source), sort_keys=True).encode()
        )
    return digest.hexdigest()[:12]


def load_chunks(
    settings: Settings, sources: tuple[PolicySource, ...]
) -> list[PolicyChunk]:
    """Chunks for `sources`, read from the on-disk cache and built on first use."""
    cache = settings.cache_dir / f"policy_chunks-{_cache_key(settings, sources)}.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [PolicyChunk.model_validate(c) for c in raw]
    chunks = build_chunks(settings, sources)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([c.model_dump() for c in chunks]), encoding="utf-8")
    logger.info(f"Built policy index: {len(chunks)} chunks")
    return chunks
