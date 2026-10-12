"""The coverage loop: bounded, deterministic stop, additive."""

import asyncio

from domain import Issue, PolicyChunk
from llm import CacheMissError, LLMClient
from policy.config import load_config
from policy.coverage import (
    MAX_ATTEMPTS,
    clean,
    coverage,
    retrieve_with_coverage,
)
from policy.index import PolicyIndex
from policy.retrieval import retrieve_policy

CONFIG = load_config()


class Scripted(LLMClient):
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def complete(  # type: ignore[override]
        self, model, system, user, images=None, max_tokens=4000, stage="llm"
    ) -> str:
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else '{"queries": []}'
        if isinstance(reply, Exception):
            raise reply
        return reply


def index() -> PolicyIndex:
    texts = [
        "Settlement hierarchy and town centres.",
        "Cesspools and septic tanks are discouraged.",
        "Wind turbines need a noise assessment.",
        "Bat roosts must be surveyed.",
    ]
    return PolicyIndex(
        [PolicyChunk(source="S", page=i + 1, text=t) for i, t in enumerate(texts)]
    )


def run(client: LLMClient, model: str | None = "m"):
    return asyncio.run(
        retrieve_with_coverage(
            client, model, index(), CONFIG, Issue.RURAL, "barn conversion"
        )
    )


def test_coverage_counts_stems_and_lists_what_is_missing() -> None:
    chunks = [PolicyChunk(source="S", page=1, text="Surface water drained safely")]
    score, missing = coverage(["surface water drains flooding"], chunks)
    assert 0 < score < 1
    assert any(m.startswith("flood") for m in missing)
    assert not any(m.startswith("drain") for m in missing)


def test_no_queries_means_full_coverage() -> None:
    assert coverage([], []) == (1.0, [])


def test_clean_drops_duplicates_and_junk() -> None:
    assert (
        clean(["Bat roosts survey", "bat   roosts survey", "x"], ["bat roosts survey"])
        == []
    )
    assert clean(["bat roosts survey"], []) == ["bat roosts survey"]


def test_reformulation_adds_extracts_and_is_bounded() -> None:
    client = Scripted(
        [
            '{"queries": ["bat roosts survey"]}',
            '{"queries": ["wind turbines noise"]}',
            '{"queries": ["never asked for"]}',
        ]
    )
    base = retrieve_policy(index(), CONFIG, Issue.RURAL, "barn conversion")
    got = run(client)
    assert client.calls <= MAX_ATTEMPTS
    assert len(got) >= len(base)
    assert {c.page for c in base} <= {c.page for c in got}  # additive only


def test_stops_when_model_proposes_nothing_new_or_fails() -> None:
    assert run(Scripted(['{"queries": []}'])) == run(None)
    assert run(Scripted([RuntimeError("down")])) == run(None)


def test_no_model_means_no_calls() -> None:
    client = Scripted([])
    run(client, None)
    assert client.calls == 0


def test_cache_miss_surfaces() -> None:
    import pytest

    with pytest.raises(CacheMissError):
        run(Scripted([CacheMissError("miss")]))
