"""Offline retrieval metrics: did the policies the officer cited reach the assessor?

Everything here works from saved run artefacts (`outputs/runs/<run>/<case>/`) and the
policy index; it never calls an LLM. The officer's citations are taken from the
reference decisions, and compared with (a) the policies in the extracts the assessor
was given, and (b) the policies our own reasons cite.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from decision import Decision
from domain import LOCAL_PLAN, Issue
from evaluator.citations import (
    SetScores,
    extract_local_plan_policies,
    extract_nppf_paragraphs,
    set_scores,
)
from pipeline.models import AssessmentBundle
from policy.config import PolicyConfig
from policy.index import PolicyIndex
from policy.retrieval import policy_block, retrieve_policy

LOCAL_PLAN_HEADER = re.compile(r"\(Policy (\d+):")
NPPF_HEADER = re.compile(r"^NPPF p\.\d+ \(paragraphs? (\d+)(?:-(\d+))?\)")

# Variant name -> (use_curated, use_search) for `retrieve_policy`.
# name -> (curated policies, keyword search, hybrid fusion instead of BM25 alone)
VARIANTS: dict[str, tuple[bool, bool, bool]] = {
    "curated+search": (True, True, True),
    "curated+bm25": (True, True, False),
    "search-only": (False, True, True),
    "bm25-only": (False, True, False),
    "curated-only": (True, False, True),
}


def local_plan_in(citations: Iterable[str]) -> set[int]:
    """Local Plan policy numbers named in retrieved-chunk citations."""
    numbers: set[int] = set()
    for citation in citations:
        if citation.startswith(LOCAL_PLAN):
            match = LOCAL_PLAN_HEADER.search(citation)
            if match:
                numbers.add(int(match.group(1)))
    return numbers


def nppf_covered(citations: Iterable[str]) -> set[int]:
    """NPPF paragraph numbers covered by the paragraph ranges of retrieved chunks."""
    covered: set[int] = set()
    for citation in citations:
        match = NPPF_HEADER.match(citation)
        if match:
            low = int(match.group(1))
            high = int(match.group(2) or low)
            covered.update(range(low, high + 1))
    return covered


@dataclass(frozen=True)
class OfficerCitations:
    """Policies the officer's reference decision relies on."""

    local_plan: frozenset[int]
    nppf: frozenset[int]


def officer_citations(reference: Decision) -> OfficerCitations:
    """Local Plan policies from reasons and conditions; NPPF paragraphs from reasons."""
    reasons = " ".join(reference.reasons)
    everything = f"{reasons} {' '.join(reference.conditions)}"
    return OfficerCitations(
        frozenset(extract_local_plan_policies(everything)),
        frozenset(extract_nppf_paragraphs(reasons)),
    )


@dataclass(frozen=True)
class RetrievalScore:
    """Recall of the officer's citations in one retrieved set, with its size."""

    case_id: str
    variant: str
    cited: frozenset[int]
    retrieved: frozenset[int]
    missed: frozenset[int]
    recall: float
    nppf_cited: frozenset[int]
    nppf_covered: frozenset[int]
    chunks: int
    chars: int | None


def score_retrieval(
    case_id: str,
    variant: str,
    officer: OfficerCitations,
    citations: list[str],
    chars: int | None,
) -> RetrievalScore:
    """Score a retrieved set (as citation strings) against the officer's citations.

    Args:
        case_id: Case being scored.
        variant: Name of the retrieval variant (curated, search, ...).
        officer: Policies the reference decision cites.
        citations: Citation strings of everything retrieved for the case.
        chars: Total characters of retrieved text, if known.
    """
    retrieved = local_plan_in(citations)
    cited = set(officer.local_plan)
    missed = cited - retrieved
    return RetrievalScore(
        case_id=case_id,
        variant=variant,
        cited=frozenset(cited),
        retrieved=frozenset(retrieved),
        missed=frozenset(missed),
        recall=len(cited & retrieved) / len(cited) if cited else 0.0,
        nppf_cited=officer.nppf,
        nppf_covered=frozenset(nppf_covered(citations) & officer.nppf),
        chunks=len(citations),
        chars=chars,
    )


def citations_of(bundle: AssessmentBundle) -> list[str]:
    """Every chunk citation the assessor was given across all issues."""
    return [c for cites in bundle.citations.values() for c in cites]


def replay_retrieval(
    index: PolicyIndex, config: PolicyConfig, bundle: AssessmentBundle, variant: str
) -> tuple[list[str], int]:
    """Re-run retrieval for a saved case under `variant`.

    Args:
        index: Policy index to search.
        config: Issue recipes.
        bundle: Saved case; its issues, facts and constraints drive the replay.
        variant: Key of `VARIANTS` selecting curated and/or keyword retrieval,
            and whether search is the hybrid fusion or BM25 alone.

    Returns:
        The citations retrieved across all issues and the total characters of text.
    """
    use_curated, use_search, hybrid = VARIANTS[variant]
    citations: list[str] = []
    chars = 0
    for assessment in bundle.assessments:
        chunks = retrieve_policy(
            index,
            config,
            Issue(assessment.issue),
            bundle.facts.development_type,
            bundle.constraints,
            use_curated=use_curated,
            use_search=use_search,
            hybrid=hybrid,
        )
        citations += [c.citation for c in chunks]
        chars += len(policy_block(chunks))  # exactly what the prompt contains
    return citations, chars


def reasons_overlap(reference: Decision, predicted: Decision) -> SetScores:
    """Overlap of Local Plan policies cited in our reasons vs the officer's reasons."""
    return set_scores(
        extract_local_plan_policies(" ".join(reference.reasons)),
        extract_local_plan_policies(" ".join(predicted.reasons)),
    )
