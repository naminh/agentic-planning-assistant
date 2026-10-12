"""Helpers for diagnosing a pipeline run against its reference decision.

The supplied evaluator reports only counts. These helpers keep the judge's full
alignment (which reason matched which, at what similarity), cache it on disk so
re-scoring is free and stable, score conditions with the same judge, link each
miss back to pipeline evidence, and compare runs.
"""

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ValidationError

from config import Settings
from decision import Decision
from domain import Finding, Issue
from evaluator.evaluator import PlanningDecisionSimpleEvaluator
from evaluator.prompt import JUDGEMENT_PROMPT_TEMPLATE
from evaluator.scores import (
    MatchedPair,
    ReasoningAlignment,
    ReasoningAlignmentScores,
)
from pipeline.models import AssessmentBundle, IssueAssessment
from pipeline.run import Runner
from policy.config import PolicyConfig, load_config
from policy.index import tokenize

STAGES = (
    "extraction",
    "routing",
    "retrieval",
    "assessment",
    "synthesis",
    "wording",
    "judge-error",
)
SCOPES = ("general", "case-specific")
METRICS = ("precision", "recall", "f1")
WEAK_SIMILARITY = 0.65
DEV_CASES = [f"case-{n:03d}" for n in range(1, 6)]
DEFAULT_BASELINE = "baseline"
MAX_JUDGE_TOKENS = 8000


# --------------------------------------------------------------------------- runs


@dataclass
class RunCase:
    """Saved artefacts of one case in one run."""

    case_id: str
    run: str | None
    decision: Decision
    bundle: AssessmentBundle | None
    brief: str
    trace: list[dict[str, object]] = field(default_factory=list)


def case_path(settings: Settings, case_id: str, run: str | None = None) -> Path:
    """Folder holding a case's artefacts (`run=None` is the latest copy)."""
    if run is None:
        return Path(settings.outputs_dir / case_id)
    return Path(settings.runs_dir / run / case_id)


def list_runs(settings: Settings) -> list[str]:
    """Names of saved run snapshots, oldest name first."""
    if not settings.runs_dir.exists():
        return []
    return sorted(p.name for p in settings.runs_dir.iterdir() if p.is_dir())


def load_run(settings: Settings, case_id: str, run: str | None = None) -> RunCase:
    """Load decision, bundle, brief and trace (each optional except the decision)."""
    folder = case_path(settings, case_id, run)
    decision = Decision.model_validate_json(
        (folder / "decision.json").read_text(encoding="utf-8")
    )
    bundle_file, brief_file, trace_file = (
        folder / "bundle.json",
        folder / "brief.md",
        folder / "trace.jsonl",
    )
    bundle = (
        AssessmentBundle.model_validate_json(bundle_file.read_text(encoding="utf-8"))
        if bundle_file.exists()
        else None
    )
    brief = brief_file.read_text(encoding="utf-8") if brief_file.exists() else ""
    trace = (
        [json.loads(line) for line in trace_file.read_text().splitlines() if line]
        if trace_file.exists()
        else []
    )
    return RunCase(case_id, run, decision, bundle, brief, trace)


def load_reference(settings: Settings, case_id: str) -> Decision | None:
    """Ground-truth decision for a eval case, or None for test cases."""
    path = settings.data_dir / "doncaster" / "decisions" / case_id / "decision.json"
    if not path.exists():
        return None
    return Decision.model_validate_json(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ scoring


def resolve_index(text: str, items: list[str]) -> int | None:
    """Index of the item closest to `text` (judges sometimes lightly reword)."""
    if not items:
        return None
    ratios = [SequenceMatcher(None, text, item).ratio() for item in items]
    best = max(range(len(items)), key=lambda i: ratios[i])
    return best if ratios[best] >= 0.6 else None


class RobustEvaluator(PlanningDecisionSimpleEvaluator):
    """The supplied evaluator, with an output-token cap and retry on truncation.

    Long reason lists (stronger models write more) can make the judge's JSON
    overrun the provider's default output limit and arrive cut off.
    """

    async def _judge_reasoning_alignment(
        self, expected: list[str], generated: list[str], retries: int = 4
    ) -> ReasoningAlignment:
        for attempt in range(retries + 1):
            try:
                resp = await self.client.responses.parse(  # type: ignore[union-attr]
                    model=self.model,
                    instructions="You compare planning reasons semantically.",
                    input=JUDGEMENT_PROMPT_TEMPLATE.render(
                        real=expected, generated=generated
                    ),
                    text_format=ReasoningAlignment,
                    max_output_tokens=MAX_JUDGE_TOKENS,
                )
            except ValidationError as err:  # truncated / malformed JSON
                logger.warning(f"judge invalid (try {attempt + 1}): {err}")
                continue
            if resp.output_parsed is not None:
                return resp.output_parsed
        raise ValueError("Judge returned no valid alignment after retries")


class CachedJudge:
    """The supplied alignment judge, with an on-disk cache and empty-list handling."""

    def __init__(
        self, evaluator: PlanningDecisionSimpleEvaluator, cache_dir: Path
    ) -> None:
        self._evaluator = evaluator
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, expected: list[str], generated: list[str], sample: int) -> str:
        blob = json.dumps([self._evaluator.model, sample, expected, generated])
        return hashlib.sha256(blob.encode()).hexdigest()

    async def align(
        self, expected: list[str], generated: list[str], sample: int = 0
    ) -> ReasoningAlignment:
        """Align two lists; `sample` > 0 forces an independent re-judgement."""
        if not expected or not generated:
            return ReasoningAlignment(
                missing_in_generation=list(expected),
                extra_in_generation=list(generated),
            )
        path = self._cache_dir / f"{self._key(expected, generated, sample)}.json"
        if path.exists():
            return ReasoningAlignment.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        # Same call the official evaluator makes, so the numbers are comparable.
        alignment = await self._evaluator._judge_reasoning_alignment(
            expected, generated
        )
        path.write_text(alignment.model_dump_json(indent=1), encoding="utf-8")
        return alignment


@dataclass
class KindResult:
    """Alignment and scores for one list kind (reasons or conditions)."""

    kind: str
    expected: list[str]
    generated: list[str]
    alignments: list[ReasoningAlignment]
    scores: list[ReasoningAlignmentScores]

    @property
    def primary(self) -> ReasoningAlignmentScores:
        """Scores from the first judge sample."""
        return self.scores[0]

    @property
    def alignment(self) -> ReasoningAlignment:
        """The first judge sample's alignment (the one the tables show)."""
        return self.alignments[0]

    def band(self, metric: str) -> tuple[float, float]:
        """(min, max) of a metric across judge samples."""
        values = [float(getattr(s, metric)) for s in self.scores]
        return min(values), max(values)


@dataclass
class CaseScore:
    """Everything the scorecard needs for one case."""

    case_id: str
    decision_match: bool
    expected_decision: str
    predicted_decision: str
    reasons: KindResult
    conditions: KindResult | None


async def score_kind(
    judge: CachedJudge,
    kind: str,
    expected: list[str],
    generated: list[str],
    samples: int = 1,
) -> KindResult:
    """Judge one list kind `samples` times (samples are cached independently).

    Args:
        judge: Cached judge to align the lists with.
        kind: "reasons" or "conditions" (a label for the result).
        expected: Reference items.
        generated: Our items.
        samples: Independent judge runs; more than one shows judge noise.
    """
    alignments = [await judge.align(expected, generated, i) for i in range(samples)]
    scores = [ReasoningAlignmentScores.from_alignment(a) for a in alignments]
    return KindResult(kind, expected, generated, alignments, scores)


async def score_case(
    judge: CachedJudge,
    case_id: str,
    expected: Decision,
    predicted: Decision,
    samples: int = 1,
) -> CaseScore:
    """Score reasons (official metric) and conditions (our extension).

    Conditions are None when neither side has any, which makes P/R/F1 undefined.

    Args:
        judge: Cached judge to align the lists with.
        case_id: Case being scored.
        expected: Reference decision.
        predicted: Our decision.
        samples: Independent judge runs per list.
    """
    reasons = await score_kind(
        judge, "reasons", expected.reasons, predicted.reasons, samples
    )
    conditions = None
    if expected.conditions or predicted.conditions:
        conditions = await score_kind(
            judge, "conditions", expected.conditions, predicted.conditions, samples
        )
    return CaseScore(
        case_id,
        expected.decision == predicted.decision,
        expected.decision.value,
        predicted.decision.value,
        reasons,
        conditions,
    )


def scorecard(scores: list[CaseScore]) -> pd.DataFrame:
    """One row per case: decision match plus reasons and conditions P/R/F1."""
    rows = []
    for s in scores:
        row: dict[str, object] = {
            "case": s.case_id,
            "expected": s.expected_decision,
            "predicted": s.predicted_decision,
            "match": int(s.decision_match),
        }
        for prefix, kind in (("", s.reasons), ("cond_", s.conditions)):
            for metric in METRICS:
                row[prefix + metric] = (
                    getattr(kind.primary, metric) if kind else float("nan")
                )
            for count in ("tp", "fp", "fn"):
                row[prefix + count] = getattr(kind.primary, count) if kind else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("case")


def alignment_rows(result: KindResult) -> pd.DataFrame:
    """Matched, missing and extra items as one table (first judge sample)."""
    rows: list[dict[str, object]] = []
    for pair in result.alignment.matched:
        rows.append(_pair_row(pair, result))
    for text in result.alignment.missing_in_generation:
        rows.append(
            {
                "status": "missing",
                "similarity": "",
                "score": 0.0,
                "reference": text,
                "generated": "",
                "ref_idx": resolve_index(text, result.expected),
                "gen_idx": None,
            }
        )
    for text in result.alignment.extra_in_generation:
        rows.append(
            {
                "status": "extra",
                "similarity": "",
                "score": 0.0,
                "reference": "",
                "generated": text,
                "ref_idx": None,
                "gen_idx": resolve_index(text, result.generated),
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("ref_idx", "gen_idx"):
        frame[column] = frame[column].astype("Int64")
    return frame


def _pair_row(pair: MatchedPair, result: KindResult) -> dict[str, object]:
    return {
        "status": "weak" if pair.score < WEAK_SIMILARITY else "matched",
        "similarity": pair.similarity.value,
        "score": pair.score,
        "reference": pair.real,
        "generated": pair.generated,
        "ref_idx": resolve_index(pair.real, result.expected),
        "gen_idx": resolve_index(pair.generated, result.generated),
    }


# --------------------------------------------------------------- evidence linking


@dataclass
class IssueEvidence:
    """How strongly an item relates to one planning issue, and what the run did."""

    issue: str
    overlap: float
    routed: bool
    finding: str | None
    substantive: bool | None
    confidence: str | None
    citations: list[str]


def _overlap(item_tokens: set[str], other: str) -> float:
    if not item_tokens:
        return 0.0
    return len(item_tokens & set(tokenize(other))) / len(item_tokens)


def _assessment_text(a: IssueAssessment) -> str:
    return " ".join(
        [
            a.issue.value.replace("_", " "),
            a.rationale,
            *a.evidence,
            *a.policy_refs,
            *a.proposed_conditions,
        ]
    )


def link_evidence(
    text: str,
    bundle: AssessmentBundle,
    top: int = 3,
    config: PolicyConfig | None = None,
) -> list[IssueEvidence]:
    """Rank issues by lexical overlap with `text`, flagging ones never routed.

    A strong match on an unrouted issue points at a routing failure; a routed
    issue with weak citations points at retrieval; a routed issue that found
    the right thing but the item is absent from the output points at synthesis.

    Args:
        text: A reference or generated reason/condition to explain.
        bundle: Saved assessments, facts and citations for the case.
        top: Number of best-matching issues to return.
        config: Policy config; loaded when None.
    """
    config = config or load_config()
    tokens = set(tokenize(text))
    routed = {a.issue: a for a in bundle.assessments}
    evidence: list[IssueEvidence] = []
    for issue in Issue:
        assessment = routed.get(issue)
        if assessment:
            overlap = _overlap(tokens, _assessment_text(assessment))
        else:
            recipe = config.recipes.get(issue)
            vocabulary = " ".join(
                [
                    issue.value.replace("_", " "),
                    recipe.label if recipe else "",
                    *list(config.routing.hints_for(issue)),
                ]
            )
            overlap = _overlap(tokens, vocabulary)
        if overlap == 0:
            continue
        evidence.append(
            IssueEvidence(
                issue=issue.value,
                overlap=round(overlap, 3),
                routed=assessment is not None,
                finding=assessment.finding.value if assessment else None,
                substantive=assessment.substantive if assessment else None,
                confidence=assessment.confidence if assessment else None,
                citations=bundle.citations.get(issue.value, []),
            )
        )
    return sorted(evidence, key=lambda e: e.overlap, reverse=True)[:top]


def link_facts(text: str, bundle: AssessmentBundle, top: int = 3) -> list[str]:
    """Extracted facts (flags, measurements, consultee notes) closest to `text`."""
    tokens = set(tokenize(text))
    facts = bundle.facts
    candidates = [
        *(f"flag: {f}" for f in facts.flags),
        *(f"measurement: {m}" for m in facts.key_measurements),
        *(f"missing: {m}" for m in facts.missing_information),
        *(f"{c.source} ({c.position}): {c.summary}" for c in facts.consultees),
    ]
    scored = sorted(
        ((_overlap(tokens, c), c) for c in candidates), key=lambda p: p[0], reverse=True
    )
    return [c for score, c in scored[:top] if score > 0]


def conflicts(bundle: AssessmentBundle) -> list[IssueAssessment]:
    """Assessments that found a conflict, substantive ones first."""
    found = [a for a in bundle.assessments if a.finding == Finding.CONFLICTS]
    return sorted(found, key=lambda a: not a.substantive)


# ------------------------------------------------------------------------- trace


def trace_frame(run_case: RunCase) -> pd.DataFrame:
    """One row per LLM call: stage, model, cache hit, cost and sizes."""
    rows = [
        {
            "call": i,
            "stage": t["stage"],
            "model": t["model"],
            "cached": t["cached"],
            "cost_usd": t["cost_usd"],
            "images": t["n_images"],
            "prompt_chars": len(str(t["system"])) + len(str(t["user"])),
            "reply_chars": len(str(t["reply"])),
        }
        for i, t in enumerate(run_case.trace)
    ]
    return pd.DataFrame(rows)


def policy_seen(run_case: RunCase, issue: str) -> str:
    """The POLICY EXTRACTS block exactly as the assessor saw it (from the trace)."""
    for entry in run_case.trace:
        if entry["stage"] == f"assess:{issue}":
            match = re.search(
                r"POLICY EXTRACTS:\n(.*?)\n\nAPPLICATION PACK:",
                str(entry["user"]),
                re.DOTALL,
            )
            return match.group(1) if match else ""
    return ""


# -------------------------------------------------------------------- run diffs


def compare_kind(a: KindResult, b: KindResult) -> pd.DataFrame:
    """Per reference item: how well each run covered it, and the delta."""

    def coverage(result: KindResult) -> dict[int, tuple[float, str]]:
        found: dict[int, tuple[float, str]] = {}
        for pair in result.alignment.matched:
            idx = resolve_index(pair.real, result.expected)
            if idx is not None:
                found[idx] = (pair.score, pair.generated)
        return found

    cov_a, cov_b = coverage(a), coverage(b)
    rows = []
    for i, text in enumerate(a.expected):
        score_a, gen_a = cov_a.get(i, (0.0, ""))
        score_b, gen_b = cov_b.get(i, (0.0, ""))
        rows.append(
            {
                "reference": text,
                "score_a": score_a,
                "score_b": score_b,
                "delta": round(score_b - score_a, 2),
                "generated_a": gen_a,
                "generated_b": gen_b,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ labels


class Label(BaseModel):
    """A manual failure diagnosis for one reference or generated item."""

    case: str
    kind: str = "reasons"
    status: str = "missing"
    item: str
    stage: str
    scope: str = "general"
    note: str = ""


def labels_path(settings: Settings, case_id: str) -> Path:
    """Where manual labels for a case are stored."""
    return Path(settings.outputs_dir / "labels" / f"{case_id}.json")


def load_labels(settings: Settings, case_id: str) -> list[Label]:
    """Saved labels for a case (empty if none)."""
    path = labels_path(settings, case_id)
    if not path.exists():
        return []
    return [Label.model_validate(x) for x in json.loads(path.read_text("utf-8"))]


def save_label(settings: Settings, label: Label) -> None:
    """Insert or replace the label for (case, kind, item)."""
    if label.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if label.scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    path = labels_path(settings, label.case)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [
        x
        for x in load_labels(settings, label.case)
        if (x.kind, x.item) != (label.kind, label.item)
    ]
    kept.append(label)
    path.write_text(
        json.dumps([x.model_dump() for x in kept], indent=1), encoding="utf-8"
    )


def label_summary(settings: Settings, cases: list[str]) -> pd.DataFrame:
    """Failure counts by stage and scope across cases (the tuning to-do list)."""
    rows = [lab.model_dump() for c in cases for lab in load_labels(settings, c)]
    if not rows:
        return pd.DataFrame(columns=["stage", "scope", "count"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["stage", "scope"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )


# ------------------------------------------------------ re-running and baselines


@dataclass
class RerunReport:
    """What a batch re-run did: cost, cache use and any per-case failures."""

    run: str
    cases: list[str]
    api_calls: int
    cached_calls: int
    cost_usd: float
    errors: dict[str, str]


def meta_path(settings: Settings, run: str) -> Path:
    """Where a run records the configuration it used."""
    return Path(settings.runs_dir / run / "meta.json")


def write_meta(settings: Settings, run: str, extra: dict[str, object]) -> None:
    """Record models and run facts so A/B comparisons can be checked for fairness."""
    meta = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "cheap_model": settings.cheap_model,
        "strong_model": settings.strong_model,
        **extra,
    }
    path = meta_path(settings, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load_meta(settings: Settings, run: str) -> dict[str, object]:
    """A run's recorded configuration (empty if it predates meta.json)."""
    path = meta_path(settings, run)
    if not path.exists():
        return {}
    loaded: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def model_mismatch(settings: Settings, run_a: str, run_b: str) -> list[str]:
    """Human-readable warnings when two runs used different models (or are unknown)."""
    meta_a, meta_b = load_meta(settings, run_a), load_meta(settings, run_b)
    warnings = []
    for key in ("cheap_model", "strong_model"):
        a, b = meta_a.get(key), meta_b.get(key)
        if a is None or b is None:
            warnings.append(f"{key}: unknown for {run_a if a is None else run_b}")
        elif a != b:
            warnings.append(f"{key} differs: {run_a}={a} vs {run_b}={b}")
    return warnings


def settings_for_models(settings: Settings, models: str) -> Settings:
    """Settings that run every stage on one chosen tier.

    `"cheap"` -> the cheap model everywhere; `"strong"` -> the configured tiers
    (strong model for vision, assessment and synthesis); anything else is treated
    as a model id used for every stage.
    """
    if models == "cheap":
        return settings.all_cheap()
    if models == "strong":
        return settings
    return replace(settings, cheap_model=models, strong_model=models)


async def rerun_cases(
    settings: Settings,
    cases: list[str],
    run: str,
    *,
    cache_only: bool = False,
    overwrite: bool = False,
) -> RerunReport:
    """Assess `cases` concurrently into `outputs/runs/<run>/`, leaving "latest" alone.

    Pass already-adjusted settings (e.g. `settings.all_cheap()`); what is passed is
    what runs and what gets recorded in meta.json. Failed cases are reported, not
    raised, so the others are still saved.

    Args:
        settings: Models and paths for this run.
        cases: Case ids to assess.
        run: Run name; results go to `outputs/runs/<run>/`.
        cache_only: Replay from the LLM cache and fail on any miss.
        overwrite: Allow reusing an existing run name.
    """
    if (settings.runs_dir / run).exists() and not overwrite:
        raise ValueError(f"Run {run!r} already exists; pick a new name or overwrite")
    runner = await Runner.create(settings, cache_only=cache_only)
    results = await asyncio.gather(
        *(runner.assess(c) for c in cases), return_exceptions=True
    )
    errors: dict[str, str] = {}
    for case_id, result in zip(cases, results, strict=True):
        if isinstance(result, BaseException):
            errors[case_id] = f"{type(result).__name__}: {result}"
            continue
        result.save(settings, run=run, latest=False)
    usage = runner.client.usage
    report = RerunReport(
        run,
        [c for c in cases if c not in errors],
        usage.calls,
        usage.cached,
        round(usage.cost_usd, 4),
        errors,
    )
    write_meta(settings, run, {"cache_only": cache_only, "report": asdict(report)})
    return report


def baseline_name(settings: Settings) -> str:
    """The run currently treated as the baseline (pointer file, default 'baseline')."""
    pointer = settings.runs_dir / "_baseline"
    if pointer.exists():
        return str(pointer.read_text(encoding="utf-8").strip())
    return DEFAULT_BASELINE


def set_baseline(settings: Settings, run: str, cases: list[str] | None = None) -> str:
    """Point the baseline at `run`; nothing is copied or deleted. Returns the old one."""
    needed = cases or DEV_CASES
    missing = [
        c
        for c in needed
        if not (case_path(settings, c, run) / "decision.json").exists()
    ]
    if missing:
        raise ValueError(f"Run {run!r} is missing decisions for {missing}")
    previous = baseline_name(settings)
    (settings.runs_dir / "_baseline").write_text(run + "\n", encoding="utf-8")
    return previous


async def score_run(
    judge: CachedJudge,
    settings: Settings,
    run: str | None,
    cases: list[str] | None = None,
    samples: int = 1,
) -> dict[str, CaseScore]:
    """Score every case of a run that has a reference and a saved decision.

    Args:
        judge: Cached judge to align the lists with.
        settings: Paths to the runs and reference decisions.
        run: Run name, or None for the latest copy.
        cases: Case ids to score; defaults to the eval cases.
        samples: Independent judge runs per list.

    Returns:
        Scores keyed by case id; cases without a reference or decision are skipped.
    """
    scored: dict[str, CaseScore] = {}
    for case_id in cases or DEV_CASES:
        reference = load_reference(settings, case_id)
        if (
            reference is None
            or not (case_path(settings, case_id, run) / "decision.json").exists()
        ):
            continue
        decision = load_run(settings, case_id, run).decision
        scored[case_id] = await score_case(judge, case_id, reference, decision, samples)
    return scored


def _noise(a: KindResult, b: KindResult) -> float:
    spread = [hi - lo for lo, hi in (a.band("f1"), b.band("f1"))]
    return round(max(spread), 4)


def ab_table(a: dict[str, CaseScore], b: dict[str, CaseScore]) -> pd.DataFrame:
    """Per-case A vs B (reasons P/R/F1, conditions F1, decision match) plus a mean row.

    `verdict` compares the reasons-F1 delta with the judge's sample-to-sample
    spread (only known when scored with several samples; with one sample the
    judge noise is invisible, so treat small deltas with care).
    """
    rows = []
    for case_id in sorted(set(a) & set(b)):
        sa, sb = a[case_id], b[case_id]
        row: dict[str, object] = {"case": case_id}
        for metric in METRICS:
            va, vb = (float(getattr(s.reasons.primary, metric)) for s in (sa, sb))
            row |= {
                f"{metric}_a": va,
                f"{metric}_b": vb,
                f"{metric}_delta": round(vb - va, 4),
            }
        ca = sa.conditions.primary.f1 if sa.conditions else float("nan")
        cb = sb.conditions.primary.f1 if sb.conditions else float("nan")
        row |= {"cond_f1_a": ca, "cond_f1_b": cb, "cond_f1_delta": round(cb - ca, 4)}
        row |= {"match_a": int(sa.decision_match), "match_b": int(sb.decision_match)}
        delta, noise = float(row["f1_delta"]), _noise(sa.reasons, sb.reasons)  # type: ignore[arg-type]
        if delta == 0:
            row["verdict"] = "same"
        elif noise > 0 and abs(delta) <= noise:
            row["verdict"] = "within noise"
        else:
            row["verdict"] = "better" if delta > 0 else "worse"
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("case")
    mean = frame.mean(numeric_only=True)
    frame.loc["MEAN"] = {**mean.to_dict(), "verdict": ""}
    return frame
