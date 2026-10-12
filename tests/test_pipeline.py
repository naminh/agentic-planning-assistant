"""Offline tests for routing, policy retrieval, outcome rules and the LLM client."""

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from config import Settings
from decision import DecisionType
from domain import Finding, Issue, PolicyChunk
from llm import LLMClient
from pipeline.decide import STANDARD_CONDITIONS, decide_outcome
from pipeline.models import CaseFacts, IssueAssessment
from pipeline.site import find_postcode, lookup_constraints
from policy.chunking import chunks_for_source
from policy.config import PolicySource, load_config
from policy.index import PolicyIndex, tokenize
from policy.retrieval import retrieve_policy
from tools.geospatial import LocationConstraints

CONFIG = load_config()


def make_facts(**overrides: object) -> CaseFacts:
    base: dict[str, object] = {
        "site_address": "1 Test Road",
        "proposal": "Erection of a dwelling",
        "development_type": "new dwelling",
    }
    return CaseFacts.model_validate({**base, **overrides})


def assessment(finding: Finding, substantive: bool) -> IssueAssessment:
    return IssueAssessment(
        issue=Issue.DESIGN, finding=finding, substantive=substantive, rationale="x"
    )


def test_unknown_postcode_is_reported_not_treated_as_unconstrained() -> None:
    site = lookup_constraints("ZZ99 9ZZ")
    assert site.constraints is None and "UNKNOWN" in site.note


def test_known_postcode_note_lists_constraints() -> None:
    site = lookup_constraints("ZZ63 7KS")
    assert site.constraints is not None and "bessacarr" in site.note


def test_find_postcode_falls_back_to_document_text() -> None:
    from pipeline.ingest import CaseDocument

    docs = [CaseDocument(name="form", text="Site: 5 Road, Doncaster ZZ63 7KS")]
    assert find_postcode(make_facts(), docs) == "ZZ63 7KS"


def test_outcome_refuses_only_on_substantive_conflict() -> None:
    assert decide_outcome([assessment(Finding.CONFLICTS, True)]) == DecisionType.REFUSE
    assert (
        decide_outcome([assessment(Finding.CONFLICTS, False)]) == DecisionType.APPROVE
    )
    assert decide_outcome([assessment(Finding.UNCLEAR, False)]) == DecisionType.APPROVE


def test_standard_conditions_cover_time_limit_and_plans() -> None:
    joined = " ".join(STANDARD_CONDITIONS).lower()
    assert "three years" in joined and "approved plans" in joined


def make_index() -> PolicyIndex:
    return PolicyIndex(
        [
            PolicyChunk(
                source="Doncaster Local Plan",
                page=143,
                policy_id="Policy 44: Residential Design",
                text="Residential amenity overlooking privacy daylight neighbours",
            ),
            PolicyChunk(
                source="Doncaster Local Plan",
                page=163,
                policy_id="Policy 57: Flood Risk Management",
                text="Flood zone sequential test exception test drainage",
            ),
        ]
    )


def test_search_ranks_relevant_chunk_and_keeps_citation() -> None:
    top = make_index().search("flood zone sequential test", k=1)[0]
    assert top.policy_id == "Policy 57: Flood Risk Management"
    assert top.citation == (
        "Doncaster Local Plan p.163 (Policy 57: Flood Risk Management)"
    )


def test_policy_text_lookup_by_prefix() -> None:
    hits = make_index().policy_text("Doncaster Local Plan", "Policy 44:")
    assert [c.page for c in hits] == [143]


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("The flood of the zone") == ["flood", "zone"]


class Echo(BaseModel):
    value: int


class FakeClient(LLMClient):
    """LLMClient whose `complete` returns scripted replies (no network)."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    async def complete(  # type: ignore[override]
        self,
        model: str,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        max_tokens: int = 4000,
        stage: str = "llm",
    ) -> str:
        reply = self.replies[self.calls]
        self.calls += 1
        return reply


def test_parse_retries_on_invalid_json() -> None:
    client = FakeClient(['{"value": "oops', '```json\n{"value": 3}\n```'])
    result = asyncio.run(client.parse(Echo, "m", "sys", "user"))
    assert result.value == 3 and client.calls == 2


def test_parse_gives_up_after_retries() -> None:
    client = FakeClient(["nope"] * 3)
    with pytest.raises(ValueError):
        asyncio.run(client.parse(Echo, "m", "sys", "user"))


def test_settings_all_cheap_routes_every_stage() -> None:
    settings = Settings(
        cheap_model="c",
        strong_model="s",
        judge_model="j",
        api_key="k",
        base_url="u",
        data_dir=Path("."),
    )
    cheap = settings.all_cheap()
    assert cheap.strong_model == "c" and cheap.judge_model == "c"


def test_scanned_pages_use_ocr_transcript() -> None:
    source = PolicySource(path="x.json", label="Scanned Direction", ocr=True)
    doc = {"pages": [{"page_number": 2, "blocks": [{"block_type": "image"}]}]}
    chunks = chunks_for_source(
        source, doc, {2: "Article 4 direction removes HMO rights"}
    )
    assert [(c.source, c.page) for c in chunks] == [("Scanned Direction", 2)]
    assert "HMO" in chunks[0].text


def test_scanned_pages_without_ocr_yield_nothing() -> None:
    source = PolicySource(path="x.json", label="Scanned Direction", ocr=True)
    doc = {"pages": [{"page_number": 1, "blocks": [{"block_type": "image"}]}]}
    assert chunks_for_source(source, doc, {}) == []


def test_legislation_only_searched_when_named() -> None:
    law = CONFIG.legislation[1]
    index = PolicyIndex(
        [
            PolicyChunk(
                source=law, page=1, text="development plan determination section"
            ),
            PolicyChunk(
                source="NPPF", page=1, text="development plan determination policy"
            ),
        ],
        CONFIG.opt_in_labels,
    )
    assert {c.source for c in index.search("development plan determination")} == {
        "NPPF"
    }
    assert [c.source for c in index.search("development plan", sources={law})] == [law]


def test_source_labels_are_unique() -> None:
    labels = [s.label for s in CONFIG.sources]
    assert len(labels) == len(set(labels))


def _gating_index() -> PolicyIndex:
    return PolicyIndex(
        [
            PolicyChunk(
                source="High Street Conservation Area Review",
                page=4,
                text="shopfront roller shutter conservation area character high street",
            ),
            PolicyChunk(
                source="Bessacarr Conservation Area Appraisal",
                page=2,
                text="shopfront roller shutter conservation area character bessacarr",
            ),
            PolicyChunk(
                source="Town and Country Planning Act 1990",
                page=9,
                text="determination in accordance with the development plan",
            ),
        ],
        CONFIG.opt_in_labels,
    )


def test_conservation_area_documents_follow_the_site_constraint() -> None:
    index = _gating_index()
    facts = make_facts(development_type="shopfront alteration")
    none = retrieve_policy(index, CONFIG, Issue.HERITAGE, facts.development_type)
    assert not any("Conservation Area" in c.source for c in none)
    bessacarr = retrieve_policy(
        index,
        CONFIG,
        Issue.HERITAGE,
        facts.development_type,
        LocationConstraints(conservation_area_name="bessacarr"),
    )
    assert {c.source for c in bessacarr if "Conservation" in c.source} == {
        "Bessacarr Conservation Area Appraisal"
    }


def test_principle_issue_runs_a_targeted_statute_lookup() -> None:
    chunks = retrieve_policy(
        _gating_index(), CONFIG, Issue.PRINCIPLE, "new dwelling", None
    )
    assert any(c.source == "Town and Country Planning Act 1990" for c in chunks)
