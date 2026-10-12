"""Opt-in LangSmith tracing: stage and LLM-call spans, no images, truncated text.

Enabled only when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set; otherwise
every helper is a no-op so the pipeline runs unchanged without the dependency's
network access.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from langsmith import Client
from langsmith.run_helpers import trace
from langsmith.run_trees import RunTree

MAX_TEXT = 1500

RunType = Literal["chain", "llm"]

# Name of the experiment/run being executed (set by the CLI entry points).
run_name: ContextVar[str | None] = ContextVar("run_name", default=None)


def enabled() -> bool:
    """True when LangSmith tracing is switched on and a key is present."""
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def clip(text: str, limit: int = MAX_TEXT) -> str:
    """Truncate long prompt/reply text so traces stay light."""
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


@contextmanager
def span(
    name: str,
    run_type: RunType = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[RunTree | None]:
    """Open a child span (yields None when tracing is off).

    Args:
        name: Span name shown in LangSmith.
        run_type: LangSmith run type ("chain", "llm", ...).
        inputs: Inputs to record on the span.
        metadata: Extra metadata to record on the span.
    """
    if not enabled():
        yield None
        return
    with trace(name, run_type=run_type, inputs=inputs or {}, metadata=metadata) as run:
        yield run


def record_llm(
    run: RunTree | None,
    reply: str,
    cached: bool,
    cost_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Attach the reply, cache flag, tokens and OpenRouter-reported cost to a span.

    Args:
        run: The span from `span`; None (tracing off) makes this a no-op.
        reply: Model reply text (clipped before sending).
        cached: Whether the reply came from the disk cache.
        cost_usd: OpenRouter-reported cost; 0 for cached replies.
        prompt_tokens: Input tokens.
        completion_tokens: Output tokens.
    """
    if run is None:
        return
    run.add_metadata(
        {
            "cached": cached,
            "cost_usd": cost_usd,
            "usage_metadata": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "total_cost": cost_usd,
            },
        }
    )
    run.add_outputs({"reply": clip(reply)})


def flush() -> None:
    """Block until queued traces are uploaded (call before the process exits)."""
    if enabled():
        Client().flush()
