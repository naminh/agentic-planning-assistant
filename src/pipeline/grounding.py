"""Check that what the assessor says is backed by the text it was shown.

Pure code, no model calls. A claim is grounded when its quote appears (allowing for
whitespace, punctuation and small OCR slips) in the policy extracts or application
documents. A cited policy number is supported only if that policy was in the extracts.
"""

import re
from collections.abc import Iterable

from domain import LOCAL_PLAN, NPPF, PolicyChunk
from pipeline.ingest import CaseDocument
from pipeline.models import Claim, IssueAssessment

MIN_QUOTE_WORDS = 4  # shorter quotes must match exactly
SHINGLE = 3  # words per overlapping fragment for the fuzzy match
FUZZY_THRESHOLD = 0.85  # share of the quote's fragments that must be present

_LOCAL_PLAN_REF = re.compile(r"\bpolic(?:y|ies)\s+(\d{1,3})\b", re.I)
_POLICY_HEADING = re.compile(r"^\s*Policy\s+(\d{1,3})\s*:", re.I | re.M)
_NPPF_REF = re.compile(r"\bparagraphs?\s+(\d{1,3})\b|\bpara\.?\s*(\d{1,3})\b", re.I)


def normalise(text: str) -> str:
    """Lower-case and reduce to plain word tokens separated by single spaces."""
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def quote_supported(quote: str, haystack: str) -> bool:
    """True if `quote` appears in `haystack`, exactly or (for longer quotes) nearly."""
    needle = normalise(quote)
    if not needle:
        return False
    hay = normalise(haystack)
    if needle in hay:
        return True
    words = needle.split()
    if len(words) < MIN_QUOTE_WORDS:
        return False
    shingles = [
        " ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)
    ]
    present = sum(1 for s in shingles if s in hay)
    return present / len(shingles) >= FUZZY_THRESHOLD


def _policy_haystack(chunks: Iterable[PolicyChunk], label: str) -> str:
    """Text of the extracts whose source matches `label` (all extracts if it is blank)."""
    wanted = label.lower().strip()
    return "\n".join(
        c.text
        for c in chunks
        if not wanted or wanted in c.source.lower() or c.source.lower() in wanted
    )


def _application_haystack(docs: Iterable[CaseDocument], name: str) -> str:
    """Text of the named document; falls back to the whole pack if the name is unknown."""
    wanted = normalise(name)
    matching = [d for d in docs if wanted and wanted in normalise(d.name)]
    return "\n".join(d.full_text for d in (matching or list(docs)))


def ground_claim(
    claim: Claim, chunks: list[PolicyChunk], docs: list[CaseDocument]
) -> Claim:
    """Return `claim` with `grounded` set from a check of its quote."""
    if claim.source == "policy":
        found = quote_supported(
            claim.quote, _policy_haystack(chunks, claim.document)
        ) or quote_supported(claim.quote, _policy_haystack(chunks, ""))
    else:
        found = quote_supported(
            claim.quote, _application_haystack(docs, claim.document)
        )
    return claim.model_copy(update={"grounded": found})


def unsupported_refs(refs: Iterable[str], chunks: list[PolicyChunk]) -> list[str]:
    """Cited Local Plan policies or NPPF paragraphs that were not in the extracts.

    References to other documents (SPDs, legislation) are not number-checked.
    """
    shown_policies = {
        n for c in chunks if c.source == LOCAL_PLAN for n in _policy_numbers_in(c)
    }
    nppf_chunks = [c for c in chunks if c.source == NPPF]
    bad = []
    for ref in refs:
        lowered = ref.lower()
        if "nppf" in lowered or "national planning policy framework" in lowered:
            numbers = [int(a or b) for a, b in _NPPF_REF.findall(ref)]
            if any(not _paragraph_shown(n, nppf_chunks) for n in numbers):
                bad.append(ref)
        elif "local plan" in lowered or lowered.startswith("policy"):
            numbers = [int(n) for n in _LOCAL_PLAN_REF.findall(ref)]
            if any(n not in shown_policies for n in numbers):
                bad.append(ref)
    return bad


def _policy_numbers_in(chunk: PolicyChunk) -> set[int]:
    """Policies a chunk contains: its id plus any policy heading inside its text.

    Cross-references in running text ("see Policy 12") are not headings and do not count.
    """
    numbers = {int(n) for n in _LOCAL_PLAN_REF.findall(chunk.policy_id or "")}
    numbers |= {int(n) for n in _POLICY_HEADING.findall(chunk.text)}
    return numbers


def _paragraph_shown(number: int, nppf_chunks: list[PolicyChunk]) -> bool:
    """True if a shown NPPF chunk covers paragraph `number` (by id range or opener)."""
    for chunk in nppf_chunks:
        bounds = [int(n) for n in re.findall(r"\d+", chunk.policy_id or "")]
        if bounds and bounds[0] <= number <= bounds[-1]:
            return True
        if re.search(rf"(?:^|\n)\s*{number}\.\s", chunk.text):
            return True
    return False


def ground_assessment(
    assessment: IssueAssessment,
    chunks: list[PolicyChunk],
    docs: list[CaseDocument],
) -> IssueAssessment:
    """Mark each claim grounded or not, and move unsupported references aside."""
    claims = [ground_claim(c, chunks, docs) for c in assessment.claims]
    bad = unsupported_refs(assessment.policy_refs, chunks)
    kept = [r for r in assessment.policy_refs if r not in bad]
    return assessment.model_copy(
        update={
            "claims": claims,
            "policy_refs": kept,
            "unsupported_refs": [*assessment.unsupported_refs, *bad],
        }
    )
