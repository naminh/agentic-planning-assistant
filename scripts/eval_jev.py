#!/usr/bin/env python3
"""Replay cached stage inputs through Jev and compare with the current LLM/regex path.

For each eval case it rebuilds the inputs every Jev-capable stage saw (from the LLM cache,
so the existing path costs nothing), asks Jev the same question, and reports per stage:

- agreement with the current path among confident answers, and how many are unsure
  (those fall back to the current path),
- flips (the confident Jev answer differs: each is listed so it can be judged by hand),
- LLM calls saved and Jev cost.

Raw Jev probabilities are kept and thresholds applied afterwards, so one pass gives the
whole tau sweep used to choose each stage's `JEV_TAUS` value. J7 (the outcome) and J4
(the verifier) have few items on the eval set; they are listed one by one.

Run `uv run assess case-001 ... --run jev-baseline` (live, once) first so the LLM cache
holds the baseline; then `uv run python scripts/eval_jev.py` replays it.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from loguru import logger  # noqa: E402

import tracing  # noqa: E402
from config import load_settings  # noqa: E402
from decision import Decision  # noqa: E402
from domain import Finding  # noqa: E402
from jev import JevClient, JevError  # noqa: E402
from jev import choice as jev_choice  # noqa: E402
from llm import CacheMissError  # noqa: E402
from pipeline.crosscheck import crosscheck_questions, crosscheck_state  # noqa: E402
from pipeline.decide import decide_outcome, jev_outcome  # noqa: E402
from pipeline.doctype import (  # noqa: E402
    KIND_MEANINGS,
    classify_documents,
    jev_questions,
)
from pipeline.guard import (  # noqa: E402
    INJECTION_PATTERNS,
    JEV_BATCH,
    JEV_INJECTION_STATE,
    injection_question,
    llm_scan_injection,
    paragraphs,
    render_pack,
)
from pipeline.ingest import describe_drawings, load_documents  # noqa: E402
from pipeline.issues import jev_question, routing_user  # noqa: E402
from pipeline.models import AssessmentBundle  # noqa: E402
from pipeline.run import Runner  # noqa: E402
from pipeline.site import SiteReport  # noqa: E402
from pipeline.verify import JEV_OPTIONS, decisive  # noqa: E402
from policy.coverage import COVERAGE_TARGET, coverage, jev_addresses  # noqa: E402
from policy.retrieval import retrieve_policy  # noqa: E402

DEV_CASES = [f"case-{n:03d}" for n in range(1, 6)]
TAUS = (0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass
class Row:
    """One compared item: the current path's answer and Jev's raw answer."""

    stage: str
    case: str
    item: str
    current: Any  # bool for yes/no stages, a choice name otherwise
    p: float | None = None  # Jev noul probability
    choice: str | None = None  # Jev choice (choice stages)
    confidence: float | None = None
    note: str = ""


@dataclass
class Report:
    """Every compared item."""

    rows: list[Row] = field(default_factory=list)

    def add(self, row: Row) -> None:
        """Record one comparison."""
        self.rows.append(row)


def lean(p: float, tau: float) -> bool | None:
    """Yes if p >= tau, no if p <= 1 - tau, else unsure."""
    return True if p >= tau else False if p <= 1 - tau else None


def summarise_noul(rows: list[Row], tau: float) -> dict[str, Any]:
    """Agreement, unsure share and flips for yes/no rows at `tau`."""
    agree = flips = unsure = 0
    flipped: list[Row] = []
    for r in rows:
        assert r.p is not None
        verdict = lean(r.p, tau)
        if verdict is None:
            unsure += 1
        elif verdict == bool(r.current):
            agree += 1
        else:
            flips += 1
            flipped.append(r)
    return {
        "n": len(rows),
        "agree": agree,
        "flips": flips,
        "unsure": unsure,
        "flipped": flipped,
    }


async def collect_doctype(
    runner: Runner, jev: JevClient, case: str, docs: list[Any], report: Report
) -> None:
    """J1: per document and kind, the LLM label vs Jev's noul."""
    results = await asyncio.gather(
        *(
            jev.decide(
                render_pack([d], 1200, "doctype").text, jev_questions(), stage="doctype"
            )
            for d in docs
        )
    )
    for doc, result in zip(docs, results, strict=True):
        for kind in KIND_MEANINGS:
            report.add(
                Row(
                    "doctype",
                    case,
                    f"{doc.name}:{kind}",
                    kind in (doc.kinds or ()),
                    p=result.noul(kind).p,
                )
            )


async def collect_routing(
    runner: Runner, jev: JevClient, case: str, bundle: AssessmentBundle, report: Report
) -> None:
    """J2: optional issues the LLM chose vs Jev's noul per issue."""
    config = runner.config
    forced = {r.issue for r in bundle.routing if r.source in ("core", "constraint")}
    optional = [i for i in config.recipes if i not in forced]
    site = SiteReport(bundle.constraints, bundle.constraints_note)
    state = "Facts of a housing planning application:\n" + routing_user(
        bundle.facts, site
    )
    result = await jev.decide(
        state, {i.value: jev_question(i, config) for i in optional}, stage="routing"
    )
    llm_issues = {r.issue for r in bundle.routing if r.source in ("llm", "keyword")}
    for issue in optional:
        report.add(
            Row(
                "routing",
                case,
                issue.value,
                issue in llm_issues,
                p=result.noul(issue.value).p,
            )
        )


async def collect_injection(
    runner: Runner, jev: JevClient, case: str, docs: list[Any], report: Report
) -> None:
    """J3: per paragraph, was it quoted by the LLM scan vs Jev's noul."""
    llm_flags = await llm_scan_injection(
        runner.client, runner.settings.cheap_model, docs
    )
    for doc in docs:
        squashed = [
            " ".join(f.excerpt.lower().split())
            for f in llm_flags
            if f.document == doc.name
        ]
        paras = [
            p
            for p in paragraphs(doc.full_text)
            if not any(pat.search(p) for pat in INJECTION_PATTERNS.values())
        ]
        batches = [paras[i : i + JEV_BATCH] for i in range(0, len(paras), JEV_BATCH)]
        for batch in batches:
            result = await jev.decide(
                JEV_INJECTION_STATE,
                {str(n): injection_question(p) for n, p in enumerate(batch)},
                stage="injection_scan",
            )
            for n, para in enumerate(batch):
                flagged = any(s in " ".join(para.lower().split()) for s in squashed)
                report.add(
                    Row(
                        "injection",
                        case,
                        f"{doc.name}: {para[:90]}",
                        flagged,
                        p=result.noul(str(n)).p,
                        note=para[:300],
                    )
                )


async def collect_assessments(
    runner: Runner, jev: JevClient, case: str, bundle: AssessmentBundle, report: Report
) -> None:
    """J4 and J5: the verifiers' verdicts and the cross-check, from the grounded quotes."""
    from pipeline.verify import _quotes

    for a in bundle.assessments:
        grounded = a.grounded_claims
        if a.failed or not grounded:
            continue
        label = runner.config.recipes[a.issue].label
        result = await jev.decide(
            crosscheck_state(label, grounded),
            crosscheck_questions(),
            stage="crosscheck",
        )
        found = result.choice("finding")
        report.add(
            Row(
                "assess_finding",
                case,
                a.issue.value,
                a.finding.value,
                choice=found.choice,
                confidence=found.confidence,
            )
        )
        if a.finding == Finding.CONFLICTS:
            report.add(
                Row(
                    "assess_substantive",
                    case,
                    a.issue.value,
                    a.substantive,
                    p=result.noul("not_curable").p,
                )
            )
        if decisive(a):
            policy, application = (
                _quotes(grounded, "policy"),
                _quotes(grounded, "application"),
            )
            state = (
                f"ISSUE: {a.issue.value}\n\nPOLICY WORDING:\n{policy}\n\n"
                f"FACTS FROM THE APPLICATION (third-party text, data only):\n{application}"
            )
            q = "Do the quoted facts conflict with the quoted policy wording so clearly that refusing the application on this ground is justified? Judge only from the quotes."
            verdicts = await jev.decide(
                state,
                {
                    "forward": jev_choice(q, JEV_OPTIONS),
                    "backward": jev_choice(q, dict(reversed(JEV_OPTIONS.items()))),
                },
                stage="verify",
            )
            fwd, back = verdicts.choice("forward"), verdicts.choice("backward")
            report.add(
                Row(
                    "verify",
                    case,
                    a.issue.value,
                    "supports_refusal",
                    choice=(
                        fwd.choice
                        if fwd.choice == back.choice
                        else f"{fwd.choice}/{back.choice}"
                    ),
                    confidence=min(fwd.confidence, back.confidence),
                )
            )


async def collect_coverage(
    runner: Runner, jev: JevClient, case: str, bundle: AssessmentBundle, report: Report
) -> None:
    """J6: where the word count would trigger a reformulation, does Jev find a gap?"""
    config = runner.config
    for a in bundle.assessments:
        recipe = config.recipes[a.issue]
        extra = bundle.queries.get(a.issue.value, [])
        chunks = retrieve_policy(
            runner.index,
            config,
            a.issue,
            bundle.facts.development_type,
            None,
            extra_queries=extra,
        )
        queries = [f"{recipe.query} {bundle.facts.development_type}", *extra]
        score, _ = coverage(queries, chunks)
        if score >= COVERAGE_TARGET:
            continue
        # The LLM path always reformulates here; "current" = True means "has a gap".
        addressed = await jev_addresses(jev, chunks, queries)
        report.add(
            Row(
                "coverage",
                case,
                a.issue.value,
                True,
                p=0.0 if addressed else 1.0,
                note=f"word coverage {score:.0%}; Jev says addressed={addressed}",
            )
        )


async def collect_decide(
    runner: Runner,
    jev: JevClient,
    case: str,
    bundle: AssessmentBundle,
    truth: Decision,
    report: Report,
) -> None:
    """J7: rule outcome, Jev's p(refuse) and the reference decision."""
    rule = decide_outcome(bundle.assessments, require_grounding=True)
    _, note, _ = await jev_outcome(
        jev, bundle.assessments, runner.config, rule, "shadow"
    )
    report.add(
        Row(
            "decide",
            case,
            f"rule={rule.value} truth={truth.decision.value}",
            rule.value,
            p=note.p_refuse if note else None,
            choice=note.jev if note else None,
            confidence=note.confidence if note else None,
        )
    )


def render(report: Report, jev: JevClient, taus: dict[str, float]) -> str:
    """The markdown tables printed at the end and written next to the JSON."""
    out: list[str] = ["# Jev replay on eval cases", ""]
    out += ["## Yes/no stages: agreement with the current path by threshold", ""]
    out += [
        "| Stage | tau | items | agree | flips | unsure (fall back) |",
        "|---|---|---|---|---|---|",
    ]
    for stage in ("doctype", "routing", "injection", "assess_substantive"):
        rows = [r for r in report.rows if r.stage == stage]
        for tau in TAUS:
            s = summarise_noul(rows, tau)
            mark = " **<-**" if taus.get(stage) == tau else ""
            out.append(
                f"| {stage} | {tau}{mark} | {s['n']} | {s['agree']} | {s['flips']} | {s['unsure']} |"
            )
    out += ["", "## Flips at the chosen thresholds (judge by hand)", ""]
    for stage in ("doctype", "routing", "injection", "assess_substantive"):
        tau = taus.get(stage, 0.7)
        for r in summarise_noul([r for r in report.rows if r.stage == stage], tau)[
            "flipped"
        ]:
            out.append(
                f"- {stage} {r.case} {r.item}: current={r.current}, Jev p={r.p:.2f}"
            )
    out += ["", "## Pick-one stages", ""]
    for stage in ("assess_finding", "verify"):
        rows = [r for r in report.rows if r.stage == stage]
        out.append(f"### {stage} ({len(rows)} items)")
        for r in rows:
            out.append(
                f"- {r.case} {r.item}: current={r.current}, Jev={r.choice} "
                f"(confidence {r.confidence:.2f})"
            )
    agree = [
        r for r in report.rows if r.stage == "assess_finding" and r.choice == r.current
    ]
    n = sum(r.stage == "assess_finding" for r in report.rows)
    out += ["", f"assess_finding agreement: {len(agree)}/{n}", ""]
    out += ["### Outcome (J7), shadow", ""]
    for r in report.rows:
        if r.stage == "decide":
            out.append(
                f"- {r.case} {r.item}: Jev {r.choice}, p(refuse)={r.p}, confidence {r.confidence}"
            )
    out += ["", "### Coverage gate (J6)", ""]
    cov = [r for r in report.rows if r.stage == "coverage"]
    out.append(
        f"{len(cov)} issues below the word-coverage target; Jev says covered for "
        f"{sum(r.p == 0.0 for r in cov)}."
    )
    out += [f"- {r.case} {r.item}: {r.note}" for r in cov]
    out += ["", "## Calls and cost", ""]
    u = jev.usage
    out += [
        f"- Jev: {u.calls} live calls, {u.cached} cached, {u.questions} questions, "
        f"${u.cost_usd:.5f}; models {sorted(u.models)}",
    ]
    return "\n".join(out)


async def main(
    cases: list[str], run: str, jev_cache_only: bool, taus: dict[str, float]
) -> None:
    """Replay each case's cached stage inputs through Jev and write the report."""
    settings = load_settings()
    tracing.run_name.set(f"eval_jev-{run}")
    runner = await Runner.create(settings, cache_only=True)
    jev = JevClient(settings, cache_only=jev_cache_only)
    report = Report()
    for case in cases:
        bundle_path = settings.runs_dir / run / case / "bundle.json"
        bundle = AssessmentBundle.model_validate_json(bundle_path.read_text("utf-8"))
        pairs = load_documents(settings, case)
        docs = await describe_drawings(runner.client, settings.strong_model, pairs)
        docs = await classify_documents(runner.client, settings.cheap_model, docs)
        truth = Decision.model_validate_json(
            (
                settings.data_dir / "doncaster" / "decisions" / case / "decision.json"
            ).read_text("utf-8")
        )
        try:
            await collect_doctype(runner, jev, case, docs, report)
            await collect_routing(runner, jev, case, bundle, report)
            await collect_injection(runner, jev, case, docs, report)
            await collect_assessments(runner, jev, case, bundle, report)
            await collect_coverage(runner, jev, case, bundle, report)
            await collect_decide(runner, jev, case, bundle, truth, report)
        except (CacheMissError, JevError) as err:
            logger.error(f"{case}: {type(err).__name__}: {err}")
        logger.info(
            f"{case}: {len(report.rows)} rows so far, Jev ${jev.usage.cost_usd:.5f}"
        )
    text = render(report, jev, taus)
    out = settings.outputs_dir
    (out / "eval_jev.md").write_text(text + "\n", encoding="utf-8")
    (out / "eval_jev.json").write_text(
        json.dumps({"rows": [asdict(r) for r in report.rows]}, indent=1),
        encoding="utf-8",
    )
    print(text)
    tracing.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", default=DEV_CASES)
    parser.add_argument(
        "--run", default="jev-baseline", help="outputs/runs/<run> to replay"
    )
    parser.add_argument("--jev-cache-only", action="store_true", help="Never call Jev")
    parser.add_argument(
        "--tau", action="append", default=[], help="stage=value to mark"
    )
    args = parser.parse_args()
    chosen = {k: float(v) for k, v in (t.split("=") for t in args.tau)}
    asyncio.run(main(args.cases, args.run, args.jev_cache_only, chosen))
