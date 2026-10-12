"""Offline candidate generation on a tiny synthetic corpus."""

from domain import LOCAL_PLAN, NPPF, Issue, PolicyChunk
from policy.candidates import (
    apply_candidates,
    build_catalogue,
    candidate_report,
    rank_candidates,
)
from policy.config import IssueRecipe, PolicyRef
from policy.index import PolicyIndex


def corpus() -> PolicyIndex:
    return PolicyIndex(
        [
            PolicyChunk(
                source=LOCAL_PLAN,
                page=163,
                policy_id="Policy 57: Flood Risk Management",
                text="Policy 57: Flood Risk Management\nFlood zone sequential test exception test.",
            ),
            PolicyChunk(
                source=LOCAL_PLAN,
                page=143,
                policy_id="Policy 44: Residential Design",
                text="Policy 44: Residential Design\nResidential amenity overlooking privacy.",
            ),
            PolicyChunk(
                source="NPPF",
                page=49,
                text="170. Inappropriate development in areas at risk of flooding should be avoided by directing development away.",
            ),
            PolicyChunk(
                source="NPPF",
                page=39,
                text="135. Planning policies and decisions should ensure that developments function well.",
            ),
        ]
    )


RECIPE = IssueRecipe(
    label="Flood risk",
    query="flood zone sequential test drainage",
    local_plan=[PolicyRef(policy=44)],  # deliberately a poor match
    nppf=[PolicyRef(para=170)],
)


def test_catalogue_has_one_entry_per_policy_and_paragraph() -> None:
    catalogue = build_catalogue(corpus())
    assert sorted(c.policy_id or "" for c in catalogue if c.source == LOCAL_PLAN) == [
        "Policy 44: Residential Design",
        "Policy 57: Flood Risk Management",
    ]
    assert {c.policy_id for c in catalogue if c.source == NPPF} == {
        "paragraph 135",
        "paragraph 170",
    }


def test_ranking_prefers_topical_entries_and_marks_recipe_membership() -> None:
    ranked = rank_candidates(PolicyIndex(build_catalogue(corpus())), RECIPE)
    assert (
        ranked["local_plan"][0].number == 57 and not ranked["local_plan"][0].in_recipe
    )
    assert ranked["nppf"][0].number == 170 and ranked["nppf"][0].in_recipe


def test_report_flags_recipe_entries_that_rank_poorly() -> None:
    report = candidate_report(corpus(), {Issue.FLOOD: RECIPE}, top=1)
    assert "## flood_risk_and_drainage" in report
    assert "Policy 57" in report
    assert "Policy 44 shares no keywords with the query" in report


def _config() -> dict:
    return {
        "issues": {
            "flood_risk_and_drainage": {
                "label": "Flood risk",
                "query": "flood zone sequential test drainage",
                "local_plan": [{"policy": 44, "expect": "residential design"}],
                "nppf": [{"para": 135, "expect": "function well"}],
            }
        }
    }


def test_apply_candidates_adds_missing_top_entries_flagged_generated() -> None:
    config = _config()
    added = apply_candidates(config, PolicyIndex(build_catalogue(corpus())), top=1)
    assert ("flood_risk_and_drainage", "local_plan", 57) in added
    assert ("flood_risk_and_drainage", "nppf", 170) in added
    entry = config["issues"]["flood_risk_and_drainage"]
    assert entry["local_plan"][0] == {"policy": 44, "expect": "residential design"}
    assert entry["local_plan"][-1] == {"policy": 57, "generated": True}
    assert entry["nppf"][-1] == {"para": 170, "generated": True}


def test_apply_candidates_is_idempotent() -> None:
    config = _config()
    catalogue = PolicyIndex(build_catalogue(corpus()))
    apply_candidates(config, catalogue, top=1)
    snapshot = repr(config)
    assert apply_candidates(config, catalogue, top=1) == []
    assert repr(config) == snapshot
