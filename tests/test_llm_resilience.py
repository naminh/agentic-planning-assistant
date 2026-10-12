"""Offline tests for the LLM client's behaviour on rate-limited, flaky free models."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import llm
from config import Settings
from llm import LLMClient, retry_delay


def settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "cheap_model": "a/cheap:free",
        "strong_model": "a/strong:free",
        "judge_model": "a/cheap:free",
        "api_key": "k",
        "base_url": "http://x",
        "cache_dir": tmp_path / "cache",
        "max_concurrency": 2,
    }
    return Settings(**{**base, **overrides})  # type: ignore[arg-type]


def reply(text: str | None, finish: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=None)


def status_error(
    cls: type[openai.APIStatusError], code: int, retry_after: str | None = None
) -> openai.APIStatusError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(
        code, headers=headers, request=httpx.Request("POST", "http://x")
    )
    return cls("boom", response=response, body=None)


class FakeAPI:
    """Stands in for `AsyncOpenAI`: plays back scripted replies and errors."""

    def __init__(self, script: list[object]) -> None:
        self.script = list(script)
        self.calls = 0
        self.kwargs: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        self.kwargs.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        assert isinstance(step, SimpleNamespace)
        return step


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "BACKOFF_BASE_S", 0.0)


def client_with(
    tmp_path: Path, script: list[object], **kw: object
) -> tuple[LLMClient, FakeAPI]:
    kw = {"requests_per_minute": 600_000, **kw}  # tests should not wait on pacing
    client = LLMClient(settings(tmp_path, **kw))
    fake = FakeAPI(script)
    client._client = fake  # type: ignore[assignment]
    return client, fake


def test_rate_limit_is_retried_until_it_succeeds(tmp_path: Path) -> None:
    limited = status_error(openai.RateLimitError, 429, retry_after="0")
    client, fake = client_with(tmp_path, [limited, limited, reply("ok")])
    assert asyncio.run(client.complete("m", "s", "u")) == "ok"
    assert fake.calls == 3


def test_server_errors_and_timeouts_are_retried(tmp_path: Path) -> None:
    errors = [
        status_error(openai.InternalServerError, 503),
        openai.APITimeoutError(request=httpx.Request("POST", "http://x")),
        reply("fine"),
    ]
    client, fake = client_with(tmp_path, errors)
    assert asyncio.run(client.complete("m", "s", "u")) == "fine"
    assert fake.calls == 3


def test_empty_reply_is_retried_and_never_cached(tmp_path: Path) -> None:
    client, fake = client_with(tmp_path, [reply(""), reply(None), reply("answer")])
    assert asyncio.run(client.complete("m", "s", "u")) == "answer"
    assert fake.calls == 3
    cached = list((tmp_path / "cache" / "llm").glob("*.txt"))
    assert [p.read_text() for p in cached] == ["answer"]


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    bad = status_error(openai.BadRequestError, 400)
    client, fake = client_with(tmp_path, [bad, reply("never")])
    with pytest.raises(openai.BadRequestError):
        asyncio.run(client.complete("m", "s", "u"))
    assert fake.calls == 1


def test_gives_up_after_the_attempt_limit(tmp_path: Path) -> None:
    limited = status_error(openai.RateLimitError, 429, retry_after="0")
    client, fake = client_with(tmp_path, [limited] * llm.MAX_ATTEMPTS)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(client.complete("m", "s", "u"))
    assert fake.calls == llm.MAX_ATTEMPTS


def test_a_cached_reply_costs_no_request(tmp_path: Path) -> None:
    client, fake = client_with(tmp_path, [reply("once")])
    assert asyncio.run(client.complete("m", "s", "u")) == "once"
    assert asyncio.run(client.complete("m", "s", "u")) == "once"
    assert fake.calls == 1


def test_retry_after_header_is_honoured() -> None:
    assert (
        retry_delay(status_error(openai.RateLimitError, 429, retry_after="7"), 0) == 7
    )
    assert retry_delay(status_error(openai.BadRequestError, 400), 0) is None
    assert retry_delay(ValueError("unrelated"), 0) is None


def test_free_models_get_a_default_pace_and_paid_ones_do_not(tmp_path: Path) -> None:
    assert settings(tmp_path).effective_rpm == 16
    paid = settings(tmp_path, cheap_model="a/c", strong_model="a/s", judge_model="a/c")
    assert paid.effective_rpm == 0
    assert settings(tmp_path, requests_per_minute=5).effective_rpm == 5


def test_requests_are_paced_to_the_rate_limit(tmp_path: Path) -> None:
    client, _ = client_with(tmp_path, [], requests_per_minute=1200)  # 50 ms apart
    start = time.monotonic()

    async def burst() -> None:
        await asyncio.gather(*(client._pace() for _ in range(5)))

    asyncio.run(burst())
    assert time.monotonic() - start >= 0.18  # four gaps of 50 ms


def test_fallback_models_are_sent_so_openrouter_can_route_around_a_busy_one(
    tmp_path: Path,
) -> None:
    client, fake = client_with(
        tmp_path,
        [reply("a"), reply("b")],
        fallback_models=("x/text:free", "y/text:free", "z/text:free"),
        vision_fallback_models=("v/vision:free",),
    )
    asyncio.run(client.complete("p/primary:free", "s", "u"))
    extra = fake.kwargs[0]["extra_body"]
    assert isinstance(extra, dict)
    assert extra["models"] == ["p/primary:free", "x/text:free", "y/text:free"]

    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02"
        b"\xfe\xa75\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    asyncio.run(client.complete("p/primary:free", "s", "u2", images=[png]))
    extra = fake.kwargs[1]["extra_body"]
    assert isinstance(extra, dict) and extra["models"] == [
        "p/primary:free",
        "v/vision:free",
    ]


def test_no_fallbacks_means_no_models_field(tmp_path: Path) -> None:
    client, fake = client_with(tmp_path, [reply("a")])
    asyncio.run(client.complete("p/primary", "s", "u"))
    extra = fake.kwargs[0]["extra_body"]
    assert isinstance(extra, dict) and "models" not in extra


def test_empty_reply_cut_off_by_the_token_cap_is_retried_with_a_bigger_cap(
    tmp_path: Path,
) -> None:
    client, fake = client_with(tmp_path, [reply("", finish="length"), reply("seen")])
    assert asyncio.run(client.complete("m", "s", "u", max_tokens=100)) == "seen"
    assert [k["max_tokens"] for k in fake.kwargs] == [100, 200]


def test_a_second_empty_reply_makes_the_next_try_ask_a_different_model(
    tmp_path: Path,
) -> None:
    client, fake = client_with(
        tmp_path,
        [reply(""), reply(""), reply("from the second model")],
        fallback_models=("alt/model:free",),
    )
    assert (
        asyncio.run(client.complete("p/primary:free", "s", "u"))
        == "from the second model"
    )
    # The first retry keeps the model and drops thinking; only then does it rotate.
    assert [k["model"] for k in fake.kwargs] == [
        "p/primary:free",
        "p/primary:free",
        "alt/model:free",
    ]
    assert reasoning_flags(fake) == [None, {"enabled": False}, {"enabled": False}]


def test_parse_grows_the_token_cap_when_json_is_cut_off(tmp_path: Path) -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: int

    client, fake = client_with(
        tmp_path, [reply('{"answer": 4'), reply('{"answer": 4}')]
    )
    assert asyncio.run(client.parse(Out, "m", "s", "u", max_tokens=100)).answer == 4
    assert [k["max_tokens"] for k in fake.kwargs] == [100, 200]


def reasoning_flags(fake: FakeAPI) -> list[object]:
    return [k["extra_body"].get("reasoning") for k in fake.kwargs]  # type: ignore[attr-defined]


def test_first_empty_reply_retries_with_thinking_off_under_the_original_key(
    tmp_path: Path,
) -> None:
    client, fake = client_with(tmp_path, [reply(""), reply("answer")])
    assert asyncio.run(client.complete("m", "s", "u", max_tokens=100)) == "answer"
    assert reasoning_flags(fake) == [None, {"enabled": False}]
    assert client.usage.no_think_retries == 1
    key = LLMClient._cache_key("m", "s", "u", None, 100)
    assert (tmp_path / "cache" / "llm" / f"{key}.txt").read_text() == "answer"
    # A replay with the original arguments hits the cache and makes no request.
    assert asyncio.run(client.complete("m", "s", "u", max_tokens=100)) == "answer"
    assert fake.calls == 2
    cache_only = LLMClient(settings(tmp_path), cache_only=True)
    assert asyncio.run(cache_only.complete("m", "s", "u", max_tokens=100)) == "answer"


def test_the_no_thinking_retry_pauses_only_briefly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", record)
    client, _ = client_with(tmp_path, [reply(""), reply("ok")])
    asyncio.run(client.complete("m", "s", "u"))
    assert waits == [llm.EMPTY_RETRY_DELAY_S]


def test_all_empty_replies_give_up_after_the_reduced_budget(tmp_path: Path) -> None:
    client, fake = client_with(tmp_path, [reply("")] * llm.MAX_ATTEMPTS)
    with pytest.raises(llm.EmptyReplyError):
        asyncio.run(client.complete("m", "s", "u"))
    assert fake.calls == llm.MAX_EMPTY_REPLIES < llm.MAX_ATTEMPTS
    assert reasoning_flags(fake) == [None, {"enabled": False}, {"enabled": False}]
    assert not list((tmp_path / "cache" / "llm").glob("*.txt"))


def test_other_errors_keep_the_full_budget_after_an_empty_reply(
    tmp_path: Path,
) -> None:
    limited = status_error(openai.RateLimitError, 429, retry_after="0")
    client, fake = client_with(tmp_path, [reply(""), *[limited] * 5])
    with pytest.raises(openai.RateLimitError):
        asyncio.run(client.complete("m", "s", "u"))
    assert fake.calls == llm.MAX_ATTEMPTS


def test_cache_key_is_unchanged_for_requests_that_never_fail() -> None:
    import hashlib

    expect = hashlib.sha256(b"msu100").hexdigest()
    assert LLMClient._cache_key("m", "s", "u", None, 100) == expect
    assert LLMClient._cache_key("m", "s", "u", None, 100, True) != expect
