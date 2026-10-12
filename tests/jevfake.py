"""A Jev client with a scripted transport, for offline tests of every Jev stage."""

import itertools
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from config import Settings
from jev import JevClient

Rule = Callable[[Any, dict[str, Any]], float | str | tuple[str, float] | None]


def settings(tmp_path: Path, stages: str = "", **overrides: Any) -> Settings:
    """Settings that point the cache at `tmp_path`; no key or network is needed."""
    return Settings(
        cheap_model="cheap",
        strong_model="strong",
        judge_model="judge",
        api_key="k",
        base_url="u",
        cache_dir=tmp_path,
        jev_stages=frozenset(s for s in stages.split(",") if s),
        **overrides,
    )


class FakeJev(JevClient):
    """Answers every question from `rule(state, question_payload)`.

    `rule` returns a noul probability, a choice name (probability 1) or
    `(choice, confidence)`; `None` makes the whole call fail with a server error.
    """

    def __init__(
        self,
        tmp_path: Path,
        rule: Rule,
        stages: str = "doctype,routing,injection,verify,assess,coverage",
        *,
        cache_only: bool = False,
        shared_cache: bool = False,
        **overrides: Any,
    ) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._ids = itertools.count()
        self.rule = rule
        # each client gets its own cache unless a test is about sharing one
        cache = tmp_path if shared_cache else tmp_path / uuid.uuid4().hex
        super().__init__(
            settings(cache, stages, **overrides),
            cache_only=cache_only,
            transport=self._answer,
        )

    async def _answer(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        answers: dict[str, Any] = {}
        for key, q in body["questions"].items():
            value = self.rule(body["state"], {"id": key, **q})
            if value is None:
                raise RuntimeError("scripted failure")
            if q["type"] == "noul":
                assert isinstance(value, float | int)
                answers[key] = {"type": "noul", "noul": float(value)}
            else:
                name, conf = (value, 1.0) if isinstance(value, str) else value  # type: ignore[misc]
                options = list(q["criteria"])
                rest = (1 - conf) / max(len(options) - 1, 1)
                answers[key] = {
                    "type": "choice",
                    "choice": name,
                    "probabilities": {o: conf if o == name else rest for o in options},
                    "confidence": conf,
                }
        n = next(self._ids)
        return {
            "model": "typesafe/jev-test-1",
            "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.00001},
            "id": f"gen-{n}",
        }
