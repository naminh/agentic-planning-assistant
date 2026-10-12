"""Extract policy citations from decision text and score them as sets (no LLM)."""

import re
from dataclasses import dataclass

# "Policy 44", "Policies 41, 42 and 44", "Policy 13(A)", "Policy 44A and NPPF ..."
LOCAL_PLAN_CITATION = re.compile(
    r"\bPolic(?:y|ies)\s+((?:\d{1,2}[A-Za-z]?(?:\([A-Za-z0-9]\))?(?:\s*(?:,|and|&)\s*)?)+)"
)
NPPF_MENTION = re.compile(r"NPPF|National Planning Policy Framework")
PARAGRAPHS = re.compile(
    r"\bparagraphs?\s+(\d{1,3}(?:\s*[-–]\s*\d{1,3})?(?:\s*(?:,|and|&)\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)*)",
    re.IGNORECASE,
)
PARA_ABBREV = re.compile(
    r"\bparas?\.?\s+(\d{1,3}(?:\s*[-–]\s*\d{1,3})?(?:\s*(?:,|and|&)\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)*)",
    re.IGNORECASE,
)
NUMBER_OR_RANGE = re.compile(r"(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?")
MAX_RANGE = 15  # guards against "paragraphs 11-125" style ranges exploding


def extract_local_plan_policies(text: str) -> set[int]:
    """Local Plan policy numbers cited in `text` (letter/bracket suffixes dropped)."""
    numbers: set[int] = set()
    for match in LOCAL_PLAN_CITATION.finditer(text):
        numbers.update(int(n) for n in re.findall(r"(?<![\w(])\d{1,2}", match.group(1)))
    return numbers


def _expand(spec: str) -> set[int]:
    out: set[int] = set()
    for low, high in NUMBER_OR_RANGE.findall(spec):
        start = int(low)
        end = int(high) if high else start
        if 0 <= end - start <= MAX_RANGE:
            out.update(range(start, end + 1))
    return out


def extract_nppf_paragraphs(text: str) -> set[int]:
    """NPPF paragraph numbers cited near an NPPF mention.

    Handles "NPPF paragraph 135", "NPPF paragraphs 88-89", "NPPF paras 130, 176" and
    "paragraph 135 of the NPPF". Numbers refer to whichever edition the author used
    (the reference decisions mix 2021 and 2023-24 numbering).
    """
    numbers: set[int] = set()
    for mention in NPPF_MENTION.finditer(text):
        after = re.split(r"[;)]|Polic(?:y|ies)|Local Plan", text[mention.end() :])[0][
            :80
        ]
        before = text[max(0, mention.start() - 40) : mention.start()]
        for pattern, window in ((PARAGRAPHS, after), (PARA_ABBREV, after)):
            for found in pattern.finditer(window):
                numbers |= _expand(found.group(1))
        for found in PARAGRAPHS.finditer(before):
            numbers |= _expand(found.group(1))
    return numbers


@dataclass(frozen=True)
class SetScores:
    """Precision/recall/F1 of a predicted citation set against the officer's."""

    precision: float
    recall: float
    f1: float
    missing: frozenset[int]
    extra: frozenset[int]


def set_scores(expected: set[int], predicted: set[int]) -> SetScores:
    """Score `predicted` against `expected`; empty sets score 0 (not vacuously 1)."""
    hits = expected & predicted
    precision = len(hits) / len(predicted) if predicted else 0.0
    recall = len(hits) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return SetScores(
        precision,
        recall,
        f1,
        frozenset(expected - predicted),
        frozenset(predicted - expected),
    )
