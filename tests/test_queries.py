"""Offline tests for case-specific query rewriting."""

import asyncio

from domain import Issue, PolicyChunk
from llm import LLMClient
from pipeline.models import CaseFacts
from policy.config import load_config
from policy.index import PolicyIndex
from policy.queries import QueryPlan, clean_queries, expand_queries
from policy.retrieval import retrieve_policy

CONFIG = load_config()
FACTS = CaseFacts(
    site_address="1 Mill Lane",
    proposal="Conversion of a barn to a dwelling",
    development_type="barn conversion",
)


class ScriptedClient(LLMClient):
    """Returns one scripted reply (or raises) instead of calling the API."""

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    async def complete(  # type: ignore[override]
        self,
        model: str,
        system: str,
        user: str,
        images=None,
        max_tokens=4000,
        stage="llm",
    ) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_clean_queries_keeps_only_requested_issues_and_sane_queries() -> None:
    plan = QueryPlan.model_validate(
        {
            "plans": [
                {
                    "issue": "flood_risk_and_drainage",
                    "queries": [
                        "sequential test flood zone 2",
                        "x",
                        "  sequential   test flood zone 2 ",
                        "a " * 200,
                    ],
                },
                {
                    "issue": "residential_amenity",
                    "queries": ["overlooking privacy distances"],
                },
            ]
        }
    )
    got = clean_queries(plan, [Issue.FLOOD])
    assert list(got) == [Issue.FLOOD]
    assert got[Issue.FLOOD][0] == "sequential test flood zone 2"
    assert len(got[Issue.FLOOD]) <= 2 and all(len(q) <= 160 for q in got[Issue.FLOOD])


def test_expand_queries_returns_cleaned_model_output() -> None:
    reply = '{"plans": [{"issue": "countryside_and_green_belt", "queries": ["re-use of rural buildings"]}]}'
    got = asyncio.run(
        expand_queries(ScriptedClient(reply), "m", FACTS, [Issue.RURAL], CONFIG)
    )
    assert got == {Issue.RURAL: ["re-use of rural buildings"]}


def test_expand_queries_degrades_to_nothing_when_the_model_fails() -> None:
    client = ScriptedClient(RuntimeError("rate limited"))
    assert asyncio.run(expand_queries(client, "m", FACTS, [Issue.RURAL], CONFIG)) == {}
    junk = ScriptedClient("not json at all")
    assert asyncio.run(expand_queries(junk, "m", FACTS, [Issue.RURAL], CONFIG)) == {}


def test_extra_queries_pull_in_policy_the_recipe_query_misses() -> None:
    chunks = [
        PolicyChunk(source="S", page=1, text="Re-use of rural buildings is supported."),
        PolicyChunk(source="S", page=2, text="Settlement hierarchy and town centres."),
    ]
    index = PolicyIndex(chunks)
    base = retrieve_policy(
        index, CONFIG, Issue.RURAL, "barn conversion", use_curated=False, hybrid=False
    )
    widened = retrieve_policy(
        index,
        CONFIG,
        Issue.RURAL,
        "barn conversion",
        use_curated=False,
        extra_queries=["rural buildings re-use"],
    )
    assert any(c.page == 1 for c in widened)
    assert len(widened) >= len(base)
