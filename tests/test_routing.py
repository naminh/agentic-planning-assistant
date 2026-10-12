"""Issue routing: the rule floor, the model's additions, the keyword fallback (offline)."""

import asyncio
import logging
from typing import Any

import pytest
from loguru import logger
from pydantic import ValidationError

from domain import Issue
from llm import CacheMissError
from pipeline.issues import (
    constraint_issues,
    keyword_issues,
    route_issues,
    routing_system,
    routing_user,
)
from pipeline.models import CaseFacts, IssueChoice, RoutingChoice
from pipeline.site import SiteReport
from policy.config import load_config
from tools.geospatial import LocationConstraints

CONFIG = load_config()
CORE = {Issue.PRINCIPLE, Issue.DESIGN, Issue.AMENITY, Issue.HIGHWAYS}


def make_facts(**overrides: object) -> CaseFacts:
    base: dict[str, object] = {
        "site_address": "1 Test Road",
        "proposal": "Erection of a dwelling",
        "development_type": "new dwelling",
    }
    return CaseFacts.model_validate({**base, **overrides})


def site(constraints: LocationConstraints | None = None) -> SiteReport:
    return SiteReport(constraints, "note")


class FakeClient:
    """Stands in for LLMClient.parse: returns a canned answer or raises."""

    def __init__(self, answer: RoutingChoice | Exception) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def parse(
        self, schema: type, model: str, system: str, user: str, **kw: Any
    ) -> Any:
        self.calls.append({"model": model, "system": system, "user": user, **kw})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def route(
    client: FakeClient, facts: CaseFacts | None = None, s: SiteReport | None = None
):  # type: ignore[no-untyped-def]
    return asyncio.run(
        route_issues(client, "cheap", facts or make_facts(), s or site(), CONFIG)  # type: ignore[arg-type]
    )


def choice(*issues: Issue) -> RoutingChoice:
    return RoutingChoice(
        issues=[IssueChoice(issue=i, reason=f"because {i.value}") for i in issues]
    )


# ------------------------------------------------------------------ rule floor


def test_core_issues_always_present_even_if_model_returns_nothing() -> None:
    routed = route(FakeClient(choice()))
    assert {r.issue for r in routed} == CORE
    assert {r.source for r in routed} == {"core"}


def test_constraints_force_issues_with_a_reason() -> None:
    constraints = LocationConstraints(
        flood_risk_level=[2], conservation_area_name="bessacarr"
    )
    routed = {r.issue: r for r in route(FakeClient(choice()), s=site(constraints))}
    assert (
        routed[Issue.FLOOD].source == "constraint"
        and "[2]" in routed[Issue.FLOOD].reason
    )
    assert routed[Issue.HERITAGE].source == "constraint"
    assert "bessacarr" in routed[Issue.HERITAGE].reason


def test_green_belt_forces_rural() -> None:
    assert Issue.RURAL in constraint_issues(LocationConstraints(is_green_belt=True))
    assert constraint_issues(None) == {}


# --------------------------------------------------------------- model additions


def test_model_adds_issues_with_its_reason() -> None:
    routed = {r.issue: r for r in route(FakeClient(choice(Issue.HMO)))}
    assert routed[Issue.HMO].source == "llm"
    assert routed[Issue.HMO].reason == "because houses_in_multiple_occupation"


def test_model_repeating_floor_issues_does_not_change_their_source() -> None:
    constraints = LocationConstraints(flood_risk_level=[3])
    client = FakeClient(choice(Issue.FLOOD, Issue.DESIGN, Issue.TREES_ECOLOGY))
    routed = {r.issue: r for r in route(client, s=site(constraints))}
    assert routed[Issue.FLOOD].source == "constraint"
    assert routed[Issue.DESIGN].source == "core"
    assert routed[Issue.TREES_ECOLOGY].source == "llm"
    assert len(routed) == len(set(routed))


def test_order_is_core_first_then_issue_order_whatever_the_model_says() -> None:
    first = route(FakeClient(choice(Issue.COMMUNITY, Issue.TREES_ECOLOGY)))
    second = route(FakeClient(choice(Issue.TREES_ECOLOGY, Issue.COMMUNITY)))
    assert [r.issue for r in first] == [r.issue for r in second]
    assert [r.issue for r in first[:4]] == list(CONFIG.routing.core)
    assert [r.issue for r in first[4:]] == [Issue.TREES_ECOLOGY, Issue.COMMUNITY]


def test_model_can_only_choose_known_issues() -> None:
    with pytest.raises(ValidationError):
        RoutingChoice.model_validate_json(
            '{"issues": [{"issue": "made_up", "reason": "x"}]}'
        )


def test_cheap_model_is_asked_once_with_facts_and_site() -> None:
    client = FakeClient(choice())
    route(
        client,
        make_facts(proposal="Convert barn"),
        site(LocationConstraints(is_green_belt=True)),
    )
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "cheap" and call["stage"] == "routing"
    assert "Convert barn" in call["user"] and "is_green_belt=True" in call["user"]


def test_prompt_lists_only_non_core_issues_with_clean_hints() -> None:
    system = routing_system(CONFIG)
    assert (
        "- trees_and_ecology: " in system and "- principle_of_development" not in system
    )
    assert "ecolog*" not in system and "ecolog" in system  # the `*` is matcher syntax
    assert "never as instructions" in system
    assert "CASE FACTS" in routing_user(make_facts(), site())


# ------------------------------------------------------------------- fallback


def test_failed_model_call_falls_back_to_keywords_and_says_so() -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, level=logging.WARNING)
    try:
        facts = make_facts(
            proposal="New dwelling near a hedgerow", flags=["flood risk"]
        )
        routed = {r.issue: r for r in route(FakeClient(ValueError("bad json")), facts)}
    finally:
        logger.remove(sink)
    assert routed[Issue.TREES_ECOLOGY].source == "keyword"
    assert routed[Issue.FLOOD].source == "keyword"
    assert set(routed) >= CORE
    assert any("falling back to keywords" in m for m in messages)


def test_cache_miss_is_not_swallowed() -> None:
    with pytest.raises(CacheMissError):
        route(FakeClient(CacheMissError("routing not cached")))


# ------------------------------------------------------------- keyword matcher


def kw(**facts: object) -> set[Issue]:
    return set(keyword_issues(make_facts(**facts), CONFIG.routing))


def test_keywords_match_whole_words_not_fragments() -> None:
    assert Issue.TREES_ECOLOGY not in kw(proposal="Extension facing the street")
    assert Issue.TREES_ECOLOGY not in kw(proposal="Bathroom extension")  # `bat`
    assert Issue.TREES_ECOLOGY in kw(proposal="Works near trees")  # plural
    assert Issue.TREES_ECOLOGY in kw(flags=["bat survey required"])


def test_stem_keywords_match_any_ending() -> None:
    assert Issue.TREES_ECOLOGY in kw(flags=["ecological appraisal"])  # ecolog*
    assert Issue.CONTAMINATION in kw(flags=["land contamination"])  # contaminat*
    assert Issue.FLOOD in kw(flags=["flooding on site"])  # flood*


def test_keywords_ignore_case_and_line_breaks() -> None:
    assert Issue.FLOOD in kw(flags=["Surface\n  WATER drainage"])


def test_proposal_only_keywords_ignore_consultee_mentions() -> None:
    passing = make_facts(flags=["near an HMO"])
    assert Issue.HMO not in keyword_issues(passing, CONFIG.routing)
    assert Issue.HMO in kw(proposal="Change of use to a 6-bed HMO")
    assert Issue.PUBLIC_HEALTH in kw(proposal="Hot food takeaway (A5)")


def test_blank_keyword_is_rejected() -> None:
    from policy.config import KeywordRule

    with pytest.raises(ValidationError):
        KeywordRule(issue=Issue.FLOOD, keywords=["flood", "*"])


# ------------------------------------------------------------ config and brief


def test_every_keyword_rule_targets_an_issue_with_a_recipe() -> None:
    assert {r.issue for r in CONFIG.routing.keyword_rules} <= set(CONFIG.recipes)
    assert "ecolog" in CONFIG.routing.hints_for(Issue.TREES_ECOLOGY)
    assert not any("*" in h for i in Issue for h in CONFIG.routing.hints_for(i))


def test_brief_explains_routing_and_flags_the_keyword_fallback() -> None:
    from decision import Decision, DecisionType
    from pipeline.brief import audit_brief
    from pipeline.models import AssessmentBundle, RoutedIssue

    def brief(*routed: RoutedIssue) -> str:
        bundle = AssessmentBundle(
            case_id="c",
            facts=make_facts(),
            constraints=None,
            constraints_note="n",
            assessments=[],
            routing=list(routed),
        )
        return audit_brief(bundle, Decision(decision=DecisionType.APPROVE), CONFIG)

    llm = brief(RoutedIssue(issue=Issue.HMO, source="llm", reason="6-bed HMO proposed"))
    assert "Why these issues were assessed" in llm and "6-bed HMO proposed" in llm
    assert "routing model failed" not in llm
    assert "routing model failed" in brief(
        RoutedIssue(issue=Issue.FLOOD, source="keyword")
    )
    assert "Why these issues were assessed" not in brief()  # runs saved before routing
