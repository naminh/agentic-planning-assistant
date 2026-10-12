"""The Jev client: typed answers, cache, cache-only, retries, malformed replies."""

import asyncio
from pathlib import Path

import pytest
from openai import APIConnectionError

import jev as jev_module
from config import JEV_STAGES, load_settings
from jev import ChoiceAnswer, JevClient, JevError, NoulAnswer, choice, noul
from llm import CacheMissError
from tests.jevfake import FakeJev, settings

QUESTIONS = {
    "fra": noul("Is it a flood risk assessment?", yes="It is one", no="It is not"),
    "kind": choice("What is it?", {"fra": "flood", "plans": None}),
}


def rule(state, q):  # type: ignore[no-untyped-def]
    return 0.9 if q["type"] == "noul" else ("fra", 0.8)


def decide(client: JevClient, state: str = "doc", stage: str = "t"):  # type: ignore[no-untyped-def]
    return asyncio.run(client.decide(state, QUESTIONS, stage=stage))


def test_answers_are_typed_and_record_the_versioned_model(tmp_path: Path) -> None:
    client = FakeJev(tmp_path, rule)
    result = decide(client)
    assert result.noul("fra") == NoulAnswer(0.9)
    picked = result.choice("kind")
    assert isinstance(picked, ChoiceAnswer) and picked.choice == "fra"
    assert picked.confidence == 0.8
    assert result.model == "typesafe/jev-test-1"
    assert client.usage.models == {"typesafe/jev-test-1"}
    with pytest.raises(JevError):
        result.choice("fra")


def test_request_body_has_model_state_and_typed_questions(tmp_path: Path) -> None:
    client = FakeJev(tmp_path, rule)
    decide(client, "the state")
    body = client.bodies[0]
    assert body["model"] == "typesafe/jev-1.13" and body["state"] == "the state"
    assert body["questions"]["fra"]["criteria"] == {
        "true": "It is one",
        "false": "It is not",
    }
    assert list(body["questions"]["kind"]["criteria"]) == ["fra", "plans"]


def test_identical_calls_are_served_from_the_cache(tmp_path: Path) -> None:
    client = FakeJev(tmp_path, rule, shared_cache=True)
    first, second = decide(client), decide(client)
    assert len(client.bodies) == 1
    assert not first.cached and second.cached
    assert second.cost_usd == 0 and client.usage.calls == 1 and client.usage.cached == 1
    assert decide(client, "other state").cached is False
    # a new client on the same folder reuses the disk cache
    again = FakeJev(tmp_path, rule, shared_cache=True)
    assert decide(again).cached and again.bodies == []


def test_cache_only_raises_on_a_miss_and_replays_a_hit(tmp_path: Path) -> None:
    decide(FakeJev(tmp_path, rule, shared_cache=True))
    replay = FakeJev(tmp_path, rule, cache_only=True, shared_cache=True)
    assert decide(replay).cached
    with pytest.raises(CacheMissError):
        decide(replay, "never asked")


def test_a_malformed_reply_is_an_error_and_is_not_cached(tmp_path: Path) -> None:
    async def bad(body):  # type: ignore[no-untyped-def]
        return {"model": "m", "answers": {"fra": {"type": "noul", "noul": 7}}}

    client = JevClient(settings(tmp_path), transport=bad)
    with pytest.raises(JevError):
        decide(client)
    assert list((tmp_path / "jev").glob("*.json")) == []


def test_a_choice_outside_the_options_is_rejected(tmp_path: Path) -> None:
    async def bad(body):  # type: ignore[no-untyped-def]
        return {
            "model": "m",
            "answers": {
                "fra": {"type": "noul", "noul": 0.5},
                "kind": {
                    "type": "choice",
                    "choice": "invented",
                    "probabilities": {"invented": 1},
                    "confidence": 1,
                },
            },
        }

    with pytest.raises(JevError, match="unknown option"):
        decide(JevClient(settings(tmp_path), transport=bad))


def test_transient_errors_are_retried_then_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []

    async def no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(jev_module.asyncio, "sleep", no_sleep)
    inner = FakeJev(tmp_path, rule)
    attempts = {"n": 0}

    async def flaky(body):  # type: ignore[no-untyped-def]
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        return await inner._answer(body)

    client = JevClient(settings(tmp_path), transport=flaky)
    assert decide(client).noul("fra").p == 0.9
    assert attempts["n"] == 3 and len(slept) == 2


def test_a_permanent_error_is_a_jev_error_without_retry(tmp_path: Path) -> None:
    attempts = {"n": 0}

    async def broken(body):  # type: ignore[no-untyped-def]
        attempts["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(JevError):
        decide(JevClient(settings(tmp_path), transport=broken))
    assert attempts["n"] == 1


def test_noul_lean_and_confidence() -> None:
    assert NoulAnswer(0.9).lean(0.8) is True
    assert NoulAnswer(0.1).lean(0.8) is False
    assert NoulAnswer(0.5).lean(0.8) is None
    assert NoulAnswer(0.5).confidence == 0 and NoulAnswer(1.0).confidence == 1


def test_trace_records_each_call_with_its_case(tmp_path: Path) -> None:
    from llm import case_scope

    client = FakeJev(tmp_path, rule)
    token = case_scope.set("case-x")
    try:
        decide(client, stage="doctype")
    finally:
        case_scope.reset(token)
    (entry,) = client.trace_for("case-x")
    assert entry.stage == "doctype" and entry.n_questions == 2 and not entry.cached
    assert "fra" in entry.to_json()


def test_switches_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("JEV_STAGES", "JEV_DECIDE", "JEV_TAUS", "JEV_MODEL"):
        monkeypatch.delenv(name, raising=False)
    default = load_settings(require_key=False)  # Jev is on unless switched off
    assert default.jev_enabled and default.jev_stages == set(JEV_STAGES)
    assert default.jev_model == "typesafe/jev-1.13" and default.jev_decide == "on"
    monkeypatch.setenv("JEV_STAGES", "none")
    monkeypatch.setenv("JEV_DECIDE", "off")
    off = load_settings(require_key=False)
    assert not off.jev_enabled and off.jev_stages == frozenset()
    monkeypatch.setenv("JEV_STAGES", "doctype, verify")
    monkeypatch.setenv("JEV_TAUS", "verify=0.9")
    monkeypatch.setenv("JEV_DECIDE", "shadow")
    on = load_settings(require_key=False)
    assert on.jev_stages == {"doctype", "verify"} and on.jev_enabled
    assert on.jev_tau("verify") == 0.9 and on.jev_tau("doctype") == 0.7
    assert on.jev_tau("decide") == 0.9 and on.jev_decide == "shadow"


@pytest.mark.parametrize(
    ("name", "value"),
    [("JEV_STAGES", "nonsense"), ("JEV_DECIDE", "maybe"), ("JEV_TAUS", "verify=0.3")],
)
def test_bad_switches_fail_fast(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_settings(require_key=False)
