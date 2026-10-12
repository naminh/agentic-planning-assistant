"""Offline tests for the evaluation diagnostics (fake judge, no network)."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from config import Settings
from decision import Decision, DecisionType
from domain import Finding, Issue
from evaluator.diagnostics import (
    CachedJudge,
    CaseScore,
    Label,
    RunCase,
    ab_table,
    alignment_rows,
    baseline_name,
    case_path,
    compare_kind,
    label_summary,
    link_evidence,
    link_facts,
    list_runs,
    load_labels,
    load_meta,
    load_run,
    model_mismatch,
    policy_seen,
    resolve_index,
    save_label,
    score_case,
    scorecard,
    set_baseline,
    settings_for_models,
    write_meta,
)
from evaluator.evaluator import PlanningDecisionSimpleEvaluator
from evaluator.scores import MatchedPair, ReasoningAlignment, SimilarityLevel
from llm import LLMClient, TraceEntry, UsageLog, case_scope
from pipeline.models import AssessmentBundle, CaseFacts, Consultee, IssueAssessment
from pipeline.run import CaseResult
from policy.config import load_config


class CountingEvaluator(PlanningDecisionSimpleEvaluator):
    """Evaluator whose judge returns a scripted alignment and counts calls."""

    def __init__(self, alignment: ReasoningAlignment) -> None:
        super().__init__("judge", None)
        self.alignment = alignment
        self.calls = 0

    async def _judge_reasoning_alignment(
        self, expected: list[str], generated: list[str]
    ) -> ReasoningAlignment:
        self.calls += 1
        return self.alignment


def settings_in(tmp_path: Path) -> Settings:
    return Settings(
        cheap_model="c",
        strong_model="s",
        judge_model="j",
        api_key="k",
        base_url="u",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / ".cache",
        outputs_dir=tmp_path / "outputs",
        predictions_dir=tmp_path / "predictions",
    )


EXPECTED = Decision(
    decision=DecisionType.APPROVE,
    reasons=["Design is acceptable", "Parking is adequate", "No flood risk"],
    conditions=["Start within three years"],
)
PREDICTED = Decision(
    decision=DecisionType.APPROVE,
    reasons=["The design is acceptable", "Impact on trees is acceptable"],
    conditions=["Start within three years", "Materials to match"],
)
ALIGNMENT = ReasoningAlignment(
    matched=[
        MatchedPair(
            real="Design is acceptable",
            generated="The design is acceptable",
            similarity=SimilarityLevel.EXCELLENT,
        )
    ],
    missing_in_generation=["Parking is adequate", "No flood risk"],
    extra_in_generation=["Impact on trees is acceptable"],
)


def test_score_case_reproduces_official_maths_and_scores_conditions(
    tmp_path: Path,
) -> None:
    evaluator = CountingEvaluator(ALIGNMENT)
    judge = CachedJudge(evaluator, tmp_path / "cache")
    result = asyncio.run(score_case(judge, "case-x", EXPECTED, PREDICTED))
    reasons = result.reasons.primary
    # weighted_tp 0.85; generated 1 + 1 extra; expected 1 + 2 missing
    assert reasons.precision == pytest.approx(0.85 / 2, abs=1e-3)
    assert reasons.recall == pytest.approx(0.85 / 3, abs=1e-3)
    assert result.conditions is not None and result.decision_match


def test_conditions_not_applicable_when_both_empty(tmp_path: Path) -> None:
    refuse = Decision(decision=DecisionType.REFUSE, reasons=["Harm"])
    judge = CachedJudge(CountingEvaluator(ALIGNMENT), tmp_path / "cache")
    result = asyncio.run(score_case(judge, "c", refuse, refuse))
    assert result.conditions is None
    assert (
        scorecard([result]).loc["c", "cond_f1"]
        != scorecard([result]).loc["c", "cond_f1"]
    )


def test_judge_is_cached_and_samples_are_independent(tmp_path: Path) -> None:
    evaluator = CountingEvaluator(ALIGNMENT)
    judge = CachedJudge(evaluator, tmp_path / "cache")
    for _ in range(2):
        asyncio.run(judge.align(EXPECTED.reasons, PREDICTED.reasons))
    assert evaluator.calls == 1
    asyncio.run(judge.align(EXPECTED.reasons, PREDICTED.reasons, sample=1))
    assert evaluator.calls == 2


def test_empty_generation_skips_the_judge(tmp_path: Path) -> None:
    evaluator = CountingEvaluator(ALIGNMENT)
    judge = CachedJudge(evaluator, tmp_path / "cache")
    result = asyncio.run(judge.align(["a", "b"], []))
    assert evaluator.calls == 0 and result.missing_in_generation == ["a", "b"]


def test_alignment_rows_label_matched_weak_missing_extra(tmp_path: Path) -> None:
    judge = CachedJudge(CountingEvaluator(ALIGNMENT), tmp_path / "cache")
    result = asyncio.run(score_case(judge, "c", EXPECTED, PREDICTED)).reasons
    rows = alignment_rows(result)
    assert list(rows["status"]) == ["matched", "missing", "missing", "extra"]
    assert list(rows["ref_idx"][:3]) == [0, 1, 2]


def test_resolve_index_tolerates_rewording() -> None:
    items = ["Parking is adequate", "Design is acceptable"]
    assert resolve_index("Design is acceptable.", items) == 1
    assert resolve_index("zzzz", items) is None


def make_bundle() -> AssessmentBundle:
    return AssessmentBundle(
        case_id="c",
        facts=CaseFacts(
            site_address="1 Road",
            proposal="Erection of dwelling",
            development_type="new dwelling",
            flags=["gas pipeline nearby"],
            consultees=[
                Consultee(
                    source="Highways",
                    position="no objection",
                    summary="parking adequate",
                )
            ],
        ),
        constraints=None,
        constraints_note="",
        assessments=[
            IssueAssessment(
                issue=Issue.HIGHWAYS,
                finding=Finding.COMPLIES,
                rationale="Parking spaces are adequate for the dwelling",
            )
        ],
        citations={"highways_and_parking": ["Local Plan p.1"]},
    )


def test_link_evidence_flags_unrouted_issue() -> None:
    hits = link_evidence("Flood zone drainage risk assessment", make_bundle())
    assert hits[0].issue == Issue.FLOOD.value and not hits[0].routed
    parking = link_evidence("Parking spaces adequate", make_bundle())
    assert parking[0].issue == Issue.HIGHWAYS.value and parking[0].routed


def test_link_facts_finds_related_fact() -> None:
    assert link_facts("gas pipeline", make_bundle())[0].startswith("flag:")


def test_policy_seen_reads_exact_prompt_from_trace() -> None:
    trace: list[dict[str, object]] = [
        {
            "stage": "assess:highways_and_parking",
            "user": "X\n\nPOLICY EXTRACTS:\n[p1] text\n\nAPPLICATION PACK:\nrest",
        }
    ]
    run_case = RunCase("c", None, EXPECTED, None, "", trace)
    assert policy_seen(run_case, "highways_and_parking") == "[p1] text"
    assert policy_seen(run_case, "design_and_character") == ""


def test_save_case_writes_latest_and_run_with_trace(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    token = case_scope.set("c")
    try:
        entry = TraceEntry("c", "facts", "m", "sys", "usr", 0, "{}", True, 0.0)
    finally:
        case_scope.reset(token)
    result = CaseResult("c", PREDICTED, make_bundle(), load_config(), [entry])
    result.save(settings, run="r1")
    for run in (None, "r1"):
        loaded = load_run(settings, "c", run)
        assert loaded.decision == PREDICTED and loaded.bundle is not None
        assert loaded.trace[0]["stage"] == "facts"
    assert case_path(settings, "c", "r1").parent.name == "r1"


def test_trace_is_scoped_per_case() -> None:
    client = LLMClient.__new__(LLMClient)
    client.trace = []
    for case in ("a", "b", "a"):
        token = case_scope.set(case)
        client._trace("s", "m", "sys", "u", 0, "r", False, 0.0)
        case_scope.reset(token)
    assert len(client.trace_for("a")) == 2 and len(client.trace_for("b")) == 1


def test_compare_kind_reports_delta(tmp_path: Path) -> None:
    better = ReasoningAlignment(
        matched=[
            MatchedPair(real=r, generated=r, similarity=SimilarityLevel.PERFECT)
            for r in EXPECTED.reasons
        ]
    )
    a = asyncio.run(
        score_case(
            CachedJudge(CountingEvaluator(ALIGNMENT), tmp_path / "a"),
            "c",
            EXPECTED,
            PREDICTED,
        )
    ).reasons
    b = asyncio.run(
        score_case(
            CachedJudge(CountingEvaluator(better), tmp_path / "b"),
            "c",
            EXPECTED,
            PREDICTED,
        )
    ).reasons
    delta = compare_kind(a, b)
    assert list(delta["delta"]) == [0.15, 1.0, 1.0]


def test_labels_roundtrip_upsert_and_validation(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    label = Label(case="c", item="No flood risk", stage="routing", note="not routed")
    save_label(settings, label)
    save_label(settings, label.model_copy(update={"stage": "retrieval"}))
    saved = load_labels(settings, "c")
    assert len(saved) == 1 and saved[0].stage == "retrieval"
    assert label_summary(settings, ["c"]).iloc[0]["stage"] == "retrieval"
    with pytest.raises(ValueError):
        save_label(settings, label.model_copy(update={"stage": "nonsense"}))
    json.loads((tmp_path / "outputs" / "labels" / "c.json").read_text())


def test_cache_only_client_refuses_to_call_the_api(tmp_path: Path) -> None:
    from llm import CacheMissError

    client = LLMClient(settings_in(tmp_path), cache_only=True)
    with pytest.raises(CacheMissError):
        asyncio.run(client.complete("m", "s", "u", stage="facts"))


def test_prompts_are_recorded_even_on_a_cache_miss(tmp_path: Path) -> None:
    from llm import CacheMissError

    client = LLMClient(settings_in(tmp_path), cache_only=True)
    with pytest.raises(CacheMissError):
        asyncio.run(client.complete("m", "sys", "usr", max_tokens=7, stage="facts"))
    (record,) = client.prompts
    assert (record.stage, record.system, record.user) == ("facts", "sys", "usr")
    assert record.max_tokens == 7 and not record.cached


def _case_score(case: str, alignment: ReasoningAlignment, tmp: Path) -> CaseScore:
    judge = CachedJudge(CountingEvaluator(alignment), tmp)
    return asyncio.run(score_case(judge, case, EXPECTED, PREDICTED))


def test_ab_table_verdicts_and_mean(tmp_path: Path) -> None:
    perfect = ReasoningAlignment(
        matched=[
            MatchedPair(real=r, generated=r, similarity=SimilarityLevel.PERFECT)
            for r in EXPECTED.reasons
        ]
    )
    a = {
        "case-001": _case_score("case-001", ALIGNMENT, tmp_path / "a1"),
        "case-002": _case_score("case-002", ALIGNMENT, tmp_path / "a2"),
    }
    b = {
        "case-001": _case_score("case-001", perfect, tmp_path / "b1"),
        "case-002": _case_score("case-002", ALIGNMENT, tmp_path / "a2"),
    }
    table = ab_table(a, b)
    assert table.loc["case-001", "verdict"] == "better"
    assert table.loc["case-002", "verdict"] == "same"
    assert table.loc["case-001", "f1_delta"] > 0
    assert table.loc["MEAN", "f1_delta"] == pytest.approx(
        table.loc["case-001", "f1_delta"] / 2
    )
    worse = ab_table(b, a)
    assert worse.loc["case-001", "verdict"] == "worse"


def test_baseline_pointer_moves_without_copying(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    assert baseline_name(settings) == "baseline"
    with pytest.raises(ValueError):
        set_baseline(settings, "new", cases=["case-001"])
    CaseResult("case-001", PREDICTED, make_bundle(), load_config(), []).save(
        settings, run="new"
    )
    assert set_baseline(settings, "new", cases=["case-001"]) == "baseline"
    assert baseline_name(settings) == "new"
    assert list_runs(settings) == ["new"]


def test_model_mismatch_warns_when_models_differ_or_unknown(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    write_meta(settings, "a", {})
    write_meta(replace(settings, strong_model="other"), "b", {})
    warnings = model_mismatch(settings, "a", "b")
    assert len(warnings) == 1 and "strong_model differs" in warnings[0]
    assert "unknown" in model_mismatch(settings, "a", "missing")[0]


def test_rerun_cases_saves_run_not_latest_and_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluator import diagnostics

    settings = settings_in(tmp_path)
    config = load_config()

    class FakeRunner:
        client = LLMClient.__new__(LLMClient)

        @classmethod
        async def create(cls, settings: Settings, *, cache_only: bool) -> "FakeRunner":
            cls.client.usage = UsageLog()
            return cls()

        async def assess(self, case_id: str) -> CaseResult:
            if case_id == "case-002":
                raise RuntimeError("boom")
            return CaseResult(case_id, PREDICTED, make_bundle(), config, [])

    monkeypatch.setattr(diagnostics, "Runner", FakeRunner)
    report = asyncio.run(
        diagnostics.rerun_cases(settings, ["case-001", "case-002"], "exp")
    )
    assert report.cases == ["case-001"] and "boom" in report.errors["case-002"]
    assert (case_path(settings, "case-001", "exp") / "decision.json").exists()
    assert not case_path(settings, "case-001").exists()  # latest untouched
    assert load_meta(settings, "exp")["strong_model"] == "s"
    with pytest.raises(ValueError):
        asyncio.run(diagnostics.rerun_cases(settings, ["case-001"], "exp"))


def test_settings_for_models_picks_tier_or_custom_model(tmp_path: Path) -> None:
    base = settings_in(tmp_path)
    cheap = settings_for_models(base, "cheap")
    assert (cheap.cheap_model, cheap.strong_model) == ("c", "c")
    strong = settings_for_models(base, "strong")
    assert (strong.cheap_model, strong.strong_model) == ("c", "s")
    custom = settings_for_models(base, "vendor/mid-model")
    assert (custom.cheap_model, custom.strong_model) == ("vendor/mid-model",) * 2
