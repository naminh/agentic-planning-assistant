"""Runtime settings: model tiers, data locations and OpenRouter credentials."""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Cheap tier proves the plumbing; strong tier produces the final answers.
DEFAULT_CHEAP_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_STRONG_MODEL = "anthropic/claude-sonnet-5.5"
FREE_MODEL_RPM = 16  # OpenRouter's free tier allows about 20 requests a minute
FREE_MODEL_CONCURRENCY = 2
# Tried in order when a free primary model is rate-limited upstream (check
# https://openrouter.ai/models?max_price=0 if one is retired). Override with
# MODEL_FALLBACKS / MODEL_VISION_FALLBACKS (comma-separated; empty disables).
FREE_TEXT_FALLBACKS = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "apodex/apodex-1.1-mini:free",
)
# Second verifier (`pipeline.verify`), consulted only when the first rejects a refusal.
# A free model today; set MODEL_VERIFIER2 to a stronger paid model later, no code change.
DEFAULT_VERIFIER2_MODEL = FREE_TEXT_FALLBACKS[0]
FREE_VISION_FALLBACKS = ("dots-studio/dots-3-note-preview:free",)

# Jev (`jev.py`): a calibrated decision model that answers yes/no, pick-one and score
# questions with a probability. Each pipeline stage it can take over is switched on by
# name in JEV_STAGES. Unset means all of them and JEV_DECIDE=on; JEV_STAGES=none with
# JEV_DECIDE=off gives today's LLM/regex behaviour exactly.
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_STAGES = ("doctype", "routing", "injection", "verify", "assess", "coverage")
JEV_DECIDE_MODES = ("off", "shadow", "on")
JevDecide = Literal["off", "shadow", "on"]
# Confidence a Jev answer needs before it replaces the LLM/regex path for that item
# (below it the existing path runs). Tuned on eval cases 001-005 with scripts/eval_jev.py;
# override per stage with JEV_TAUS="doctype=0.8,verify=0.9".
DEFAULT_JEV_TAU = 0.7
DEFAULT_JEV_TAUS: dict[str, float] = {"decide": 0.9}


@dataclass(frozen=True)
class Settings:
    """Immutable configuration shared by every pipeline stage."""

    cheap_model: str
    strong_model: str
    judge_model: str
    api_key: str
    base_url: str
    data_dir: Path = REPO_ROOT / "data"
    cache_dir: Path = REPO_ROOT / ".cache"
    outputs_dir: Path = REPO_ROOT / "outputs"
    predictions_dir: Path = REPO_ROOT / "predictions"
    max_concurrency: int = 8
    requests_per_minute: int = 0  # 0 = pace automatically (free models) or not at all
    refine_tables: bool = False  # rebuild policy tables as markdown when indexing
    caption_figures: bool = False  # caption figures/maps of `captions: true` sources
    fallback_models: tuple[str, ...] = ()  # tried if the primary is rate-limited
    vision_fallback_models: tuple[str, ...] = ()  # same, for requests with images
    embeddings: bool = False  # add local-embedding hits to hybrid search
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    verifier2_model: str = DEFAULT_VERIFIER2_MODEL  # tie-break when verifier 1 rejects
    caption_model: str = (
        ""  # MODEL_CAPTION: model for table/figure captioning; "" = tiers
    )
    caption_reasoning_off: bool = False  # CAPTION_REASONING_OFF: no thinking for them
    jev_model: str = DEFAULT_JEV_MODEL
    jev_url: str = JEV_URL
    jev_stages: frozenset[str] = frozenset()  # stages Jev answers; empty = none
    jev_decide: JevDecide = "off"  # J7: log only (shadow) or let Jev overrule the rule
    jev_taus: dict[str, float] = field(default_factory=dict)  # per-stage thresholds

    @property
    def table_model(self) -> str:
        """Model that rebuilds policy tables (the caption model, else the cheap tier)."""
        return self.caption_model or self.cheap_model

    @property
    def figure_model(self) -> str:
        """Model that captions policy figures (the caption model, else the strong tier)."""
        return self.caption_model or self.strong_model

    @property
    def jev_enabled(self) -> bool:
        """True if any pipeline stage (or the final decision) may call Jev."""
        return bool(self.jev_stages) or self.jev_decide != "off"

    def jev_tau(self, stage: str) -> float:
        """Confidence a Jev answer needs at `stage` before it overrides the fallback."""
        return self.jev_taus.get(stage, DEFAULT_JEV_TAUS.get(stage, DEFAULT_JEV_TAU))

    @property
    def extracted_dir(self) -> Path:
        """Folder written by `uv run parser`."""
        return self.data_dir / "extracted"

    @property
    def runs_dir(self) -> Path:
        """Named run snapshots: outputs/runs/<run>/<case>/."""
        return self.outputs_dir / "runs"

    def case_dir(self, case_id: str) -> Path:
        """Extracted application pack for a case."""
        return self.extracted_dir / "doncaster" / "applications" / case_id

    @property
    def uses_free_models(self) -> bool:
        """True if any tier is an OpenRouter `:free` model (strict rate limits)."""
        models = (self.cheap_model, self.strong_model, self.judge_model)
        return any(m.endswith(":free") for m in models)

    @property
    def effective_rpm(self) -> int:
        """Requests per minute to allow: the setting, else a safe rate for free models."""
        if self.requests_per_minute:
            return self.requests_per_minute
        return FREE_MODEL_RPM if self.uses_free_models else 0

    def all_cheap(self) -> "Settings":
        """Return settings that route every stage to the cheap model (dry run)."""
        return replace(
            self,
            strong_model=self.cheap_model,
            judge_model=self.cheap_model,
            verifier2_model=self.cheap_model,
        )


def _model_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """A comma-separated model list from the environment; unset means `default`."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(m.strip() for m in raw.split(",") if m.strip())


def _jev_stages() -> frozenset[str]:
    """JEV_STAGES as a validated set: unset means every stage; empty or `none` means none."""
    names = frozenset(_model_list("JEV_STAGES", JEV_STAGES)) - {"none", "off"}
    unknown = sorted(names - set(JEV_STAGES))
    if unknown:
        raise ValueError(f"unknown JEV_STAGES {unknown}; choose from {JEV_STAGES}")
    return names


def _jev_taus() -> dict[str, float]:
    """JEV_TAUS="stage=0.8,..." as a dict (each value a probability in (0.5, 1])."""
    taus: dict[str, float] = {}
    for item in _model_list("JEV_TAUS", ()):
        stage, _, value = item.partition("=")
        tau = float(value)
        if not 0.5 < tau <= 1:
            raise ValueError(f"JEV_TAUS {item!r}: threshold must be in (0.5, 1]")
        taus[stage.strip()] = tau
    return taus


def _jev_decide() -> JevDecide:
    mode = os.getenv("JEV_DECIDE", "on").strip().lower() or "off"
    if mode not in JEV_DECIDE_MODES:
        raise ValueError(f"JEV_DECIDE must be one of {JEV_DECIDE_MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def _env(name: str, required: bool) -> str:
    return os.environ[name] if required else os.getenv(name, "")


def _flag(name: str, *, default: bool) -> bool:
    """Boolean env flag; unset gives `default`, so `0`/`false`/`no` switch a default-on stage off."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def load_settings(require_key: bool = True) -> Settings:
    """Load settings from the environment (`.env` is read if present).

    Offline tools (retrieval metrics, candidate lists) pass `require_key=False`: they
    never build an LLM client, so no API key is needed.
    """
    load_dotenv(REPO_ROOT / ".env")
    cheap = os.getenv("MODEL_CHEAP", DEFAULT_CHEAP_MODEL)
    strong = os.getenv("MODEL_STRONG", DEFAULT_STRONG_MODEL)
    judge = os.getenv("MODEL_JUDGE", cheap)
    free = any(m.endswith(":free") for m in (cheap, strong, judge))
    default_concurrency = FREE_MODEL_CONCURRENCY if free else 8
    return Settings(
        cheap_model=cheap,
        strong_model=strong,
        judge_model=judge,
        embeddings=_flag("EMBEDDINGS", default=True),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        verifier2_model=os.getenv("MODEL_VERIFIER2", DEFAULT_VERIFIER2_MODEL),
        caption_model=os.getenv("MODEL_CAPTION", ""),
        caption_reasoning_off=_flag("CAPTION_REASONING_OFF", default=False),
        jev_model=os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL),
        jev_url=os.getenv("JEV_URL", JEV_URL),
        jev_stages=_jev_stages(),
        jev_decide=_jev_decide(),
        jev_taus=_jev_taus(),
        api_key=_env("OPENROUTER_API_KEY", require_key),
        base_url=_env("OPENROUTER_BASE_URL", require_key),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", default_concurrency)),
        requests_per_minute=int(os.getenv("LLM_RPM", 0)),
        refine_tables=_flag("REFINE_TABLES", default=True),
        caption_figures=_flag("CAPTION_FIGURES", default=True),
        fallback_models=_model_list(
            "MODEL_FALLBACKS", FREE_TEXT_FALLBACKS if free else ()
        ),
        vision_fallback_models=_model_list(
            "MODEL_VISION_FALLBACKS", FREE_VISION_FALLBACKS if free else ()
        ),
    )
