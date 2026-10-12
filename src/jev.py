"""Client for Jev, a calibrated decision model (OpenRouter `/api/alpha/decisions`).

Jev is not a chat model. It takes a text `state` and a map of typed questions and
returns, for each, an answer with a probability: `noul` (yes/no, 0..1), `choice` (one of
a fixed set, with the whole distribution and a confidence) or `score`. It cannot write
prose, read images or do arithmetic, so the pipeline uses it only where an LLM or a regex
today merely picks from a fixed set.

This module mirrors `llm.py`: replies are cached on disk (keyed on model, state and
questions), `--cache-only` raises the same `CacheMissError` on a miss, retries reuse
`retry_delay`, and every call is costed, traced and kept in `JevClient.trace`. A call
that fails raises `JevError`; each pipeline stage catches it and runs its existing path,
so Jev being down or unsure never stops a case.
"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from loguru import logger
from openai import AsyncOpenAI

import tracing
from config import Settings
from llm import (
    BACKOFF_CAP_S,
    MAX_ATTEMPTS,
    REQUEST_TIMEOUT_S,
    CacheMissError,
    case_scope,
    retry_delay,
)

Instructions = str | dict[str, Any]
Transport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class JevError(RuntimeError):
    """The call failed or the reply was not the shape asked for (not cached)."""


@dataclass(frozen=True)
class Question:
    """One typed question. Build with `noul` or `choice`."""

    type: Literal["noul", "choice", "score"]
    instructions: Instructions
    criteria: dict[str, Any] | list[Any] | None = None

    def payload(self) -> dict[str, Any]:
        """The question as the API expects it."""
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


def noul(instructions: Instructions, *, yes: str, no: str) -> Question:
    """A yes/no question; `yes` and `no` spell out exactly what each answer means."""
    return Question("noul", instructions, {"true": yes, "false": no})


def choice(instructions: Instructions, options: Mapping[str, str | None]) -> Question:
    """A pick-one question. Option order is kept, so callers can shuffle it."""
    return Question("choice", instructions, dict(options))


@dataclass(frozen=True)
class NoulAnswer:
    """Probability that the answer is yes. A noul carries no separate confidence."""

    p: float

    @property
    def confidence(self) -> float:
        """Distance from a coin flip, scaled to 0..1 (0.5 -> 0, 0.95 -> 0.9)."""
        return abs(2 * self.p - 1)

    def lean(self, tau: float) -> bool | None:
        """True if yes with probability >= tau, False if no with >= tau, else None."""
        if self.p >= tau:
            return True
        if self.p <= 1 - tau:
            return False
        return None


@dataclass(frozen=True)
class ChoiceAnswer:
    """The most probable option, the whole distribution and Jev's own confidence."""

    choice: str
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer


@dataclass(frozen=True)
class JevResult:
    """Typed answers to one `decide` call, keyed by question id."""

    answers: dict[str, Answer]
    model: str  # versioned id the API reported, e.g. typesafe/jev-1.13-20260917
    cost_usd: float
    cached: bool

    def noul(self, key: str) -> NoulAnswer:
        """The answer to a noul question (JevError if it is another type)."""
        answer = self.answers[key]
        if not isinstance(answer, NoulAnswer):
            raise JevError(f"{key} is not a noul answer")
        return answer

    def choice(self, key: str) -> ChoiceAnswer:
        """The answer to a choice question (JevError if it is another type)."""
        answer = self.answers[key]
        if not isinstance(answer, ChoiceAnswer):
            raise JevError(f"{key} is not a choice answer")
        return answer


@dataclass
class JevTrace:
    """One Jev call: what was asked, what came back, what it cost."""

    case: str | None
    stage: str
    model: str
    n_questions: int
    state_chars: int
    answers: dict[str, Any]
    cached: bool
    cost_usd: float

    def to_json(self) -> str:
        """Serialise for a jev_trace.jsonl line."""
        return json.dumps(asdict(self))


@dataclass
class JevUsage:
    """Running totals across all Jev calls in a run."""

    calls: int = 0
    cached: int = 0
    questions: int = 0
    cost_usd: float = 0.0
    models: set[str] = field(default_factory=set)  # versioned ids seen


def _parse_answers(
    questions: Mapping[str, Question], reply: Mapping[str, Any]
) -> dict[str, Answer]:
    """Validate a reply against the questions asked; JevError if it does not fit."""
    raw = reply.get("answers")
    if not isinstance(raw, dict):
        raise JevError(f"reply has no answers: {str(reply)[:200]}")
    answers: dict[str, Answer] = {}
    for key, question in questions.items():
        item = raw.get(key)
        if not isinstance(item, dict) or item.get("type") != question.type:
            raise JevError(f"answer {key!r} missing or not a {question.type}")
        try:
            if question.type == "noul":
                p = float(item["noul"])
                if not 0 <= p <= 1:
                    raise ValueError(f"noul {p} outside 0..1")
                answers[key] = NoulAnswer(p)
            elif question.type == "choice":
                probabilities = {k: float(v) for k, v in item["probabilities"].items()}
                picked = str(item["choice"])
                if picked not in (question.criteria or {}):
                    raise ValueError(f"unknown option {picked!r}")
                answers[key] = ChoiceAnswer(
                    picked, probabilities, float(item["confidence"])
                )
            else:
                raise JevError("score questions are not used by this pipeline")
        except (KeyError, TypeError, ValueError, AttributeError) as err:
            raise JevError(f"answer {key!r} malformed: {err}") from err
    return answers


class JevClient:
    """Cached, traced, retrying client for the decisions endpoint."""

    def __init__(
        self,
        settings: Settings,
        cache_only: bool = False,
        transport: Transport | None = None,
    ) -> None:
        self.settings = settings
        self.cache_only = cache_only
        self.model = settings.jev_model
        self._post = transport or self._http_post
        base, _, endpoint = settings.jev_url.rstrip("/").rpartition("/")
        self._endpoint = f"/{endpoint}"
        self._api = AsyncOpenAI(
            api_key=settings.api_key or "unset",
            base_url=base,
            max_retries=0,  # retries are ours, as in llm.py
            timeout=REQUEST_TIMEOUT_S,
        )
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self._cache_dir = settings.cache_dir / "jev"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = JevUsage()
        self.trace: list[JevTrace] = []

    def active(self, stage: str) -> bool:
        """True if `stage` is switched on in JEV_STAGES."""
        return stage in self.settings.jev_stages

    def tau(self, stage: str) -> float:
        """The confidence threshold for `stage`."""
        return self.settings.jev_tau(stage)

    def trace_for(self, case_id: str) -> list[JevTrace]:
        """Calls made while assessing `case_id`, in call order."""
        return [t for t in self.trace if t.case == case_id]

    async def decide(
        self,
        state: Instructions,
        questions: Mapping[str, Question],
        stage: str = "jev",
    ) -> JevResult:
        """Ask all `questions` about `state` in one request.

        Args:
            state: The text (or JSON object) the questions are about.
            questions: Question id -> typed question; answers come back under the ids.
            stage: Label for the trace only; not part of the cache key.

        Returns:
            Typed answers, the versioned model id and the cost (0 when cached).

        Raises:
            CacheMissError: In `cache_only` mode when the call is not cached.
            JevError: When the call fails after retries or the reply is malformed.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {k: q.payload() for k, q in questions.items()},
        }
        key = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
        path = self._cache_dir / f"{key}.json"
        state_text = state if isinstance(state, str) else json.dumps(state)
        with tracing.span(
            f"jev:{stage}",
            run_type="llm",
            inputs={"model": self.model, "state": tracing.clip(state_text)},
            metadata={"ls_model_name": self.model, "stage": stage},
        ) as run:
            if path.exists():
                reply = json.loads(path.read_text(encoding="utf-8"))
                cached = True
            elif self.cache_only:
                raise CacheMissError(
                    f"{stage} ({self.model}) is not cached; refusing to call the API"
                )
            else:
                reply = await self._request(body, stage)
                cached = False
            answers = _parse_answers(questions, reply)  # JevError before caching
            if not cached:
                path.write_text(json.dumps(reply), encoding="utf-8")
            cost = 0.0 if cached else float(reply.get("usage", {}).get("cost") or 0.0)
            served = str(reply.get("model") or self.model)
            self._record(
                stage, served, len(questions), len(state_text), answers, cached, cost
            )
            tracing.record_llm(
                run, json.dumps(reply.get("answers")), cached, cost, 0, 0
            )
        return JevResult(answers, served, cost, cached)

    def _record(
        self,
        stage: str,
        model: str,
        n_questions: int,
        state_chars: int,
        answers: Mapping[str, Answer],
        cached: bool,
        cost: float,
    ) -> None:
        self.usage.calls += 0 if cached else 1
        self.usage.cached += 1 if cached else 0
        self.usage.questions += n_questions
        self.usage.cost_usd += cost
        self.usage.models.add(model)
        self.trace.append(
            JevTrace(
                case_scope.get(),
                stage,
                model,
                n_questions,
                state_chars,
                {k: asdict(a) for k, a in answers.items()},
                cached,
                cost,
            )
        )

    async def _request(self, body: dict[str, Any], stage: str) -> dict[str, Any]:
        """One POST, retried on rate limits, server errors and dropped connections."""
        for attempt in range(MAX_ATTEMPTS):
            try:
                async with self._sem:
                    reply = await self._post(body)
                if "answers" not in reply:
                    raise JevError(f"no answers in reply: {str(reply)[:200]}")
                return reply
            except JevError:
                raise
            except Exception as err:
                wait = retry_delay(err, attempt)
                if wait is None or attempt == MAX_ATTEMPTS - 1:
                    raise JevError(f"{stage}: {type(err).__name__}: {err}") from err
                logger.warning(
                    f"{stage} (jev): {type(err).__name__}; retry "
                    f"{attempt + 1}/{MAX_ATTEMPTS - 1} in {wait:.0f}s"
                )
                await asyncio.sleep(min(wait, BACKOFF_CAP_S))
        raise AssertionError("unreachable")

    async def _http_post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST to the decisions endpoint through the openai client, which raises the
        error types `retry_delay` already understands (it retries nothing itself)."""
        data = await self._api.post(self._endpoint, body=body, cast_to=object)
        if not isinstance(data, dict):
            raise JevError("reply is not a JSON object")
        return data
