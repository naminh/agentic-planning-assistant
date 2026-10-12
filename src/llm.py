"""Thin OpenRouter client: JSON-validated calls, image inputs, disk cache, cost log."""

import asyncio
import base64
import hashlib
import io
import json
import random
import re
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)
from PIL import Image
from pydantic import BaseModel, ValidationError

import tracing
from config import Settings

MAX_IMAGE_EDGE = 1600
MAX_REPLY_TOKENS = 8000  # ceiling when growing the cap for a reasoning model
MAX_ATTEMPTS = 6  # per request, across rate limits, server errors and empty replies
MAX_EMPTY_REPLIES = 3  # empty replies tolerated per request; the 2nd try thinks no more
EMPTY_RETRY_DELAY_S = (
    2.0  # short pause after an empty reply (x number of empties so far)
)
BACKOFF_BASE_S = 4.0
BACKOFF_CAP_S = 90.0
REQUEST_TIMEOUT_S = 180.0
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# Set by the pipeline so concurrent cases sharing one client keep separate traces.
case_scope: ContextVar[str | None] = ContextVar("case_scope", default=None)


class CacheMissError(RuntimeError):
    """Raised in cache-only mode when a call would have hit the API."""


class EmptyReplyError(RuntimeError):
    """The model answered with no text (common on busy free models); worth retrying."""


def retry_delay(error: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying `error`, or None if it is not retryable.

    Honours a `Retry-After` header; otherwise backs off exponentially with jitter.
    """
    if isinstance(error, APIStatusError) and not isinstance(error, RateLimitError):
        if error.status_code not in RETRY_STATUS:
            return None
    elif not isinstance(
        error, RateLimitError | APITimeoutError | APIConnectionError | EmptyReplyError
    ):
        return None
    header = None
    response = getattr(error, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
    try:
        wait = float(header) if header else None
    except ValueError:
        wait = None
    if wait is None:
        wait = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt)
        wait *= 0.75 + random.random() / 2  # jitter so parallel calls do not realign
    return min(wait, BACKOFF_CAP_S * 2)


@dataclass
class TraceEntry:
    """One LLM call: what was asked, what came back, and what it cost."""

    case: str | None
    stage: str
    model: str
    system: str
    user: str
    n_images: int
    reply: str
    cached: bool
    cost_usd: float

    def to_json(self) -> str:
        """Serialise for a trace.jsonl line."""
        return json.dumps(asdict(self))


@dataclass(frozen=True)
class PromptRecord:
    """A prompt as it was about to be sent, kept even when the call is a cache miss."""

    stage: str
    model: str
    system: str
    user: str
    n_images: int
    max_tokens: int
    cached: bool


@dataclass
class UsageLog:
    """Running token and cost totals across all calls in a run."""

    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    no_think_retries: int = 0  # requests re-sent with thinking off after an empty reply


def shrink_image(data: bytes) -> str:
    """Downscale an image and return it as a JPEG data URI."""
    with Image.open(io.BytesIO(data)) as source:
        img = source.convert("RGB")
    img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class LLMClient:
    """Async wrapper adding caching, retries on invalid JSON, and cost tracking."""

    def __init__(self, settings: Settings, cache_only: bool = False) -> None:
        self.cache_only = cache_only
        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=0,  # retries are ours: they honour Retry-After and pacing
            timeout=REQUEST_TIMEOUT_S,
        )
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self._fallbacks = settings.fallback_models
        self._vision_fallbacks = settings.vision_fallback_models
        self._min_interval = (
            60.0 / settings.effective_rpm if settings.effective_rpm else 0.0
        )
        self._pace_lock = asyncio.Lock()
        self._next_slot = 0.0
        self._cache_dir = settings.cache_dir / "llm"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = UsageLog()
        self.trace: list[TraceEntry] = []
        self.prompts: list[PromptRecord] = []  # every prompt asked, hit or miss

    def trace_for(self, case_id: str) -> list[TraceEntry]:
        """Calls made while assessing `case_id`, in call order."""
        return [t for t in self.trace if t.case == case_id]

    @property
    def raw(self) -> AsyncOpenAI:
        """Underlying OpenAI-compatible client (used by the evaluator)."""
        return self._client

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        max_tokens: int = 4000,
        stage: str = "llm",
        reasoning_off: bool = False,
    ) -> str:
        """Return the model's text reply, served from the disk cache when identical.

        Args:
            model: OpenRouter model id.
            system: System prompt.
            user: User prompt.
            images: Raw image bytes, resized before sending. Part of the cache key.
            max_tokens: Output cap, also part of the cache key. A reply cut off at
                this limit is returned and cached as is.
            stage: Label for the trace entry only; not part of the cache key.
            reasoning_off: Ask OpenRouter to disable the model's thinking (for
                reasoning models that otherwise spend the token cap and reply empty).
                Part of the cache key only when set, so existing entries stay valid.

        Returns:
            The reply text. Callers expecting JSON should use `parse`, which
            re-asks on invalid output.

        Raises:
            CacheMissError: In `cache_only` mode when the call is not cached.
        """
        n_images = len(images or [])
        with tracing.span(
            stage,
            run_type="llm",
            inputs={
                "model": model,
                "system": tracing.clip(system),
                "user": tracing.clip(user),
                "n_images": n_images,
            },
            metadata={"ls_model_name": model, "stage": stage},
        ) as run:
            text, cached, cost, p_tok, c_tok = await self._call(
                model, system, user, images, max_tokens, stage, reasoning_off
            )
            tracing.record_llm(run, text, cached, cost, p_tok, c_tok)
        return text

    async def _call(
        self,
        model: str,
        system: str,
        user: str,
        images: list[bytes] | None,
        max_tokens: int,
        stage: str,
        reasoning_off: bool = False,
    ) -> tuple[str, bool, float, int, int]:
        """Return (reply, cached, cost, prompt tokens, completion tokens)."""
        key = self._cache_key(model, system, user, images, max_tokens, reasoning_off)
        path = self._cache_dir / f"{key}.txt"
        n_images = len(images or [])
        self.prompts.append(
            PromptRecord(
                stage, model, system, user, n_images, max_tokens, path.exists()
            )
        )
        if path.exists():
            self.usage.cached += 1
            text = str(path.read_text(encoding="utf-8"))
            self._trace(stage, model, system, user, n_images, text, True, 0.0)
            return text, True, 0.0, 0, 0
        if self.cache_only:
            raise CacheMissError(
                f"{stage} ({model}) is not cached; refusing to call the API"
            )
        content: list[ChatCompletionContentPartParam] = [{"type": "text", "text": user}]
        for image in images or []:
            content.append(
                {"type": "image_url", "image_url": {"url": shrink_image(image)}}
            )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        response = await self._create(
            model, messages, max_tokens, stage, bool(images), reasoning_off
        )
        served = getattr(response, "model", None)
        if served and served != model and not served.startswith(model):
            logger.warning(f"{stage}: {model} was unavailable; answered by {served}")
        cost = self._record(model, response)
        text = str(response.choices[0].message.content or "")
        path.write_text(text, encoding="utf-8")
        self._trace(stage, model, system, user, n_images, text, False, cost)
        usage = getattr(response, "usage", None)
        return (
            text,
            False,
            cost,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    async def _pace(self) -> None:
        """Wait for this request's slot so calls stay under the requests-per-minute cap."""
        if not self._min_interval:
            return
        async with self._pace_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def _create(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int,
        stage: str,
        has_images: bool = False,
        reasoning_off: bool = False,
    ) -> "ChatCompletion":
        """One chat completion, retried on rate limits, server errors and empty replies.

        The semaphore is released while waiting to retry, so a backing-off call does not
        block the others. Nothing is written to the cache until a reply with text arrives.
        If fallback models are configured, OpenRouter tries them in order whenever the
        primary is unavailable, so one overloaded free model does not stall a run.
        """
        chain = [model, *(self._vision_fallbacks if has_images else self._fallbacks)]
        chain = list(dict.fromkeys(chain))[:3]  # OpenRouter allows a few alternatives
        start = 0  # advanced after a repeated empty reply: ask a different model
        empties = 0
        for attempt in range(MAX_ATTEMPTS):
            order = chain[start % len(chain) :] + chain[: start % len(chain)]
            extra: dict[str, object] = {"usage": {"include": True}}
            if reasoning_off or empties:
                extra["reasoning"] = {"enabled": False}
            if len(order) > 1:
                extra["models"] = order
            try:
                async with self._sem:
                    await self._pace()
                    response = await self._client.chat.completions.create(
                        model=order[0],
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0,
                        extra_body=extra,
                    )
                if (
                    not response.choices
                    or not (response.choices[0].message.content or "").strip()
                ):
                    empties += 1
                    if (
                        response.choices
                        and response.choices[0].finish_reason == "length"
                    ):
                        # The cap ran out: allow more alongside the no-thinking retry.
                        max_tokens = min(max_tokens * 2, MAX_REPLY_TOKENS)
                    if empties > 1 or reasoning_off:
                        # Thinking was already off: try a different model next.
                        start += 1
                    raise EmptyReplyError(f"{model} returned no text")
                return response
            except Exception as err:
                wait = retry_delay(err, attempt)
                if isinstance(err, EmptyReplyError):
                    if empties >= MAX_EMPTY_REPLIES:
                        raise
                    wait = EMPTY_RETRY_DELAY_S * empties
                    if not reasoning_off and empties == 1:
                        self.usage.no_think_retries += 1
                        logger.warning(
                            f"{stage} ({model}): empty reply; retrying with "
                            "thinking disabled"
                        )
                if wait is None or attempt == MAX_ATTEMPTS - 1:
                    raise
                logger.warning(
                    f"{stage} ({model}): {type(err).__name__}; retry "
                    f"{attempt + 1}/{MAX_ATTEMPTS - 1} in {wait:.0f}s"
                )
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    async def parse[T: BaseModel](
        self,
        schema: type[T],
        model: str,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        max_tokens: int = 4000,
        retries: int = 2,
        stage: str = "llm",
    ) -> T:
        """Return a validated `schema` instance, re-asking on malformed JSON.

        Args:
            schema: Pydantic model the reply must validate against; its JSON Schema
                is appended to the system prompt.
            model: OpenRouter model id.
            system: System prompt (without the schema instruction).
            user: User prompt.
            images: Raw image bytes sent with the prompt.
            max_tokens: Output cap per attempt.
            retries: Extra attempts after an invalid reply; each re-ask includes
                the validation error.
            stage: Trace label only.

        Returns:
            A validated `schema` instance.

        Raises:
            ValueError: If no valid reply arrives after the retries.
        """
        schema_text = json.dumps(schema.model_json_schema())
        full_system = (
            f"{system}\n\nReply with a single JSON object (no prose, no code "
            f"fences) matching this JSON Schema:\n{schema_text}"
        )
        prompt = user
        limit = max_tokens
        for attempt in range(retries + 1):
            reply = await self.complete(
                model, full_system, prompt, images, limit, stage
            )
            try:
                return schema.model_validate_json(JSON_FENCE.sub("", reply).strip())
            except ValidationError as err:
                logger.warning(f"{schema.__name__} invalid (try {attempt + 1}): {err}")
                if any(e["type"] == "json_invalid" for e in err.errors()):
                    limit = min(limit * 2, MAX_REPLY_TOKENS)  # most likely cut off
                prompt = f"{user}\n\nYour previous reply was invalid: {err}\nFix it."
        raise ValueError(f"Could not obtain valid {schema.__name__} from {model}")

    def _trace(
        self,
        stage: str,
        model: str,
        system: str,
        user: str,
        n_images: int,
        reply: str,
        cached: bool,
        cost: float,
    ) -> None:
        self.trace.append(
            TraceEntry(
                case_scope.get(),
                stage,
                model,
                system,
                user,
                n_images,
                reply,
                cached,
                cost,
            )
        )

    def _record(self, model: str, response: object) -> float:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0
        cost = float((getattr(usage, "model_extra", None) or {}).get("cost") or 0.0)
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.completion_tokens += usage.completion_tokens
        self.usage.cost_usd += cost
        self.usage.by_model[model] = self.usage.by_model.get(model, 0.0) + cost
        return cost

    @staticmethod
    def _cache_key(
        model: str,
        system: str,
        user: str,
        images: list[bytes] | None,
        max_tokens: int,
        reasoning_off: bool = False,
    ) -> str:
        digest = hashlib.sha256()
        for part in (model, system, user, str(max_tokens)):
            digest.update(part.encode())
        for image in images or []:
            digest.update(hashlib.sha256(image).digest())
        if reasoning_off:
            digest.update(b"reasoning-off")
        return digest.hexdigest()
