"""Tracing helpers: opt-in gating, truncation and the LLM-call span."""

import pytest

import tracing


def test_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert not tracing.enabled()


def test_enabled_needs_flag_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert not tracing.enabled()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert tracing.enabled()


def test_span_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    with tracing.span("stage") as run:
        assert run is None
    tracing.record_llm(None, "reply", True, 0.0, 0, 0)  # must not raise
    tracing.flush()


def test_clip_truncates_long_text() -> None:
    assert tracing.clip("short") == "short"
    clipped = tracing.clip("x" * 5000)
    assert len(clipped) < 1600 and clipped.endswith("[5000 chars]")
