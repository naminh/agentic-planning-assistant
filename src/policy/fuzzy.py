"""A second, typo- and inflection-tolerant ranking that complements BM25.

BM25 matches whole words, so "drainage" does not match "drained" and a hyphenated or
OCR-split word is lost. This index scores chunks on stemmed words plus character
trigrams, with sublinear TF-IDF and cosine similarity. It is local (no API calls), so it
is free, deterministic and works offline; it is not semantic, so synonyms still depend
on the query rewriting in `policy.queries`.
"""

import math
from collections import Counter
from collections.abc import Callable, Iterable

SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s", "ly")
MIN_STEM = 4
TRIGRAM_MIN_LEN = 5  # shorter words are matched whole, not by fragments
TRIGRAM_WEIGHT = 0.5  # fragments are weaker evidence than a whole stemmed word


def stem(word: str) -> str:
    """Strip one common English suffix, keeping at least `MIN_STEM` letters."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


def features(tokens: Iterable[str]) -> dict[str, float]:
    """Stemmed words and character trigrams (prefixed `#`) for a token stream."""
    out: dict[str, float] = {}
    for token in tokens:
        root = stem(token)
        out[root] = out.get(root, 0.0) + 1
        if len(root) >= TRIGRAM_MIN_LEN:
            for i in range(len(root) - 2):
                gram = f"#{root[i : i + 3]}"
                out[gram] = out.get(gram, 0.0) + TRIGRAM_WEIGHT
    return out


class FuzzyIndex:
    """Inverted index of L2-normalised sublinear TF-IDF vectors over chunk texts."""

    def __init__(self, texts: list[str], tokenize: Callable[[str], list[str]]) -> None:
        self._tokenize = tokenize
        counts = [features(tokenize(t)) for t in texts]
        df: Counter[str] = Counter()
        for count in counts:
            df.update(count.keys())
        n = len(texts)
        self._idf = {f: math.log(1 + n / d) for f, d in df.items()}
        self._postings: dict[str, list[tuple[int, float]]] = {}
        for i, count in enumerate(counts):
            weights = {
                f: (1 + math.log(c)) * self._idf[f] if c >= 1 else c * self._idf[f]
                for f, c in count.items()
            }
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            for f, w in weights.items():
                self._postings.setdefault(f, []).append((i, w / norm))

    def scores(self, query: str) -> dict[int, float]:
        """Cosine similarity of `query` to every chunk sharing at least one feature."""
        q = features(self._tokenize(query))
        weights = {
            f: ((1 + math.log(c)) if c >= 1 else c) * self._idf[f]
            for f, c in q.items()
            if f in self._idf
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        scores: dict[int, float] = {}
        for f, qw in weights.items():
            for i, w in self._postings[f]:
                scores[i] = scores.get(i, 0.0) + (qw / norm) * w
        return scores
