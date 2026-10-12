"""Citation extraction and offline retrieval metrics (no LLM, no data files)."""

from pathlib import Path

import pytest

from decision import Decision, DecisionType
from domain import LOCAL_PLAN, Finding, Issue, PolicyChunk
from evaluator.citations import (
    extract_local_plan_policies,
    extract_nppf_paragraphs,
    set_scores,
)
from evaluator.retrieval import (
    local_plan_in,
    nppf_covered,
    officer_citations,
    reasons_overlap,
    replay_retrieval,
    score_retrieval,
)
from pipeline.models import AssessmentBundle, CaseFacts, IssueAssessment
from policy.config import load_config
from policy.index import PolicyIndex

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Policies 41, 42 and 44", {41, 42, 44}),
        ("Doncaster Local Plan Policy 13(A) and NPPF paragraph 135", {13}),
        ("Policy 44A", {44}),
        ("contrary to Policy 24(E), Policy 50", {24, 50}),
        ("NPPF paragraph 135 only", set()),
    ],
)
def test_extract_local_plan_policies(text: str, expected: set[int]) -> None:
    assert extract_local_plan_policies(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("NPPF paragraph 135", {135}),
        ("NPPF paragraphs 88-89", {88, 89}),
        ("NPPF paras 130, 176 and Policy 9", {130, 176}),
        ("accords with paragraph 135 of the NPPF", {135}),
        ("Policy 44 only", set()),
        ("NPPF paragraphs 11-125", set()),  # implausible span is ignored
    ],
)
def test_extract_nppf_paragraphs(text: str, expected: set[int]) -> None:
    assert extract_nppf_paragraphs(text) == expected


def test_set_scores_does_not_reward_empty_sets() -> None:
    assert set_scores(set(), set()).f1 == 0.0
    scores = set_scores({1, 2, 3}, {2, 3, 4})
    assert (scores.precision, scores.recall) == (
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    )
    assert scores.missing == {1} and scores.extra == {4}


def test_reference_decisions_cite_what_we_think_they_cite() -> None:
    path = REPO / "data/doncaster/decisions/case-004/decision.json"
    if not path.exists():
        pytest.skip("reference decisions not present")
    officer = officer_citations(Decision.model_validate_json(path.read_text("utf-8")))
    assert officer.local_plan == {13, 25, 41, 57}


def test_local_plan_in_reads_citation_headers() -> None:
    cites = [
        "Doncaster Local Plan p.143 (Policy 44: Residential Design (Strategic Policy))",
        "Doncaster Local Plan p.5",
        "NPPF p.39 (paragraphs 134-135)",
    ]
    assert local_plan_in(cites) == {44}
    assert nppf_covered(cites) == {134, 135}


def test_score_retrieval_reports_missed_policies() -> None:
    decision = Decision(
        decision=DecisionType.APPROVE,
        reasons=["ok under Policies 44 and 57 and NPPF paragraph 135"],
        conditions=["x"],
    )
    cites = [
        "Doncaster Local Plan p.143 (Policy 44: Residential Design)",
        "NPPF p.39 (paragraphs 134-135)",
    ]
    score = score_retrieval("case-x", "as run", officer_citations(decision), cites, 10)
    assert score.recall == 0.5 and score.missed == {57}
    assert score.nppf_covered == {135}


def test_reasons_overlap_uses_reasons_only() -> None:
    ref = Decision(
        decision=DecisionType.APPROVE, reasons=["Policy 44"], conditions=["Policy 99"]
    )
    ours = Decision(
        decision=DecisionType.APPROVE, reasons=["Policies 44 and 45"], conditions=[]
    )
    scores = reasons_overlap(ref, ours)
    assert scores.recall == 1.0 and scores.extra == {45}


def _bundle() -> AssessmentBundle:
    return AssessmentBundle(
        case_id="case-x",
        facts=CaseFacts(
            site_address="1 Road", proposal="p", development_type="extension"
        ),
        constraints=None,
        constraints_note="n",
        assessments=[
            IssueAssessment(
                issue=Issue.AMENITY, finding=Finding.COMPLIES, rationale="r"
            )
        ],
        citations={},
    )


def _index() -> PolicyIndex:
    return PolicyIndex(
        [
            PolicyChunk(
                source=LOCAL_PLAN,
                page=143,
                policy_id="Policy 44: Residential Design",
                text="residential amenity overlooking privacy",
            ),
            PolicyChunk(
                source=LOCAL_PLAN,
                page=200,
                policy_id="Policy 66: Development Viability",
                text="overlooking privacy daylight space standards garden size amenity extension",
            ),
        ]
    )


def test_ablation_variants_switch_curated_and_search_independently() -> None:
    both, _ = replay_retrieval(_index(), load_config(), _bundle(), "curated+search")
    curated, _ = replay_retrieval(_index(), load_config(), _bundle(), "curated-only")
    search, _ = replay_retrieval(_index(), load_config(), _bundle(), "search-only")
    assert local_plan_in(curated) == {44}  # only the recipe's named policies
    assert 66 in local_plan_in(search)  # keyword hit that no recipe names
    assert local_plan_in(both) == local_plan_in(curated) | local_plan_in(search)
