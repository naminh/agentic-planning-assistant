"""Run cases through the pipeline and persist the artefacts."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

import tracing
from config import Settings
from decision import Decision
from jev import JevClient, JevTrace
from llm import LLMClient, TraceEntry, case_scope
from pipeline.assess import CaseContext, assess_issues
from pipeline.brief import audit_brief
from pipeline.crosscheck import jev_crosscheck
from pipeline.decide import decide_outcome, jev_outcome, reason_review, synthesise
from pipeline.doctype import classify_documents
from pipeline.facts import extract_facts
from pipeline.guard import semantic_scan_injection
from pipeline.ingest import describe_drawings, load_documents
from pipeline.issues import route_issues
from pipeline.models import AssessmentBundle
from pipeline.review import build_review
from pipeline.site import check_site
from pipeline.verify import rejected_conflicts, verify_decisive
from policy.config import PolicyConfig, load_config
from policy.index import PolicyIndex
from policy.queries import expand_queries


@dataclass
class CaseResult:
    """Everything one assessed case produced."""

    case_id: str
    decision: Decision
    bundle: AssessmentBundle
    config: PolicyConfig
    trace: list[TraceEntry]
    jev_trace: list[JevTrace] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """LLM and Jev spend for this case (cached calls cost nothing)."""
        return sum(t.cost_usd for t in self.trace) + sum(
            t.cost_usd for t in self.jev_trace
        )

    def save(
        self,
        settings: Settings,
        *,
        run: str | None = None,
        latest: bool = True,
        prediction: bool = False,
        label: str | None = None,
    ) -> None:
        """Write decision.json, bundle.json, brief.md and trace.jsonl.

        Args:
            settings: Paths for the outputs, runs and predictions folders.
            run: If set, also keep a copy under `outputs/runs/<run>/<label or case>/`
                so runs can be compared later.
            latest: Write to `outputs/<label or case>/`.
            prediction: Also write `predictions/<case>-prediction.json`, the
                example prediction file.
            label: Folder name to use instead of the case id (e.g. for ablations).
        """
        name = label or self.case_id
        folders = [settings.outputs_dir / name] if latest else []
        if run:
            folders.append(settings.runs_dir / run / name)
        for folder in folders:
            self._write_artefacts(folder)
        if prediction:
            settings.predictions_dir.mkdir(exist_ok=True)
            path = settings.predictions_dir / f"{self.case_id}-prediction.json"
            path.write_text(self.decision.model_dump_json(indent=2) + "\n", "utf-8")

    def _write_artefacts(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        files = {
            "decision.json": self.decision.model_dump_json(indent=2),
            "bundle.json": self.bundle.model_dump_json(indent=2),
            "brief.md": audit_brief(self.bundle, self.decision, self.config),
            "trace.jsonl": "".join(t.to_json() + "\n" for t in self.trace),
        }
        if self.jev_trace:
            files["jev_trace.jsonl"] = "".join(
                t.to_json() + "\n" for t in self.jev_trace
            )
        for name, text in files.items():
            (folder / name).write_text(text, encoding="utf-8")


@dataclass
class Runner:
    """The shared pieces needed to assess cases: settings, policy config, LLM, index."""

    settings: Settings
    config: PolicyConfig
    client: LLMClient
    index: PolicyIndex
    jev: JevClient | None = None  # None unless JEV_STAGES / JEV_DECIDE switch Jev on

    @classmethod
    async def create(cls, settings: Settings, *, cache_only: bool = False) -> "Runner":
        """Build the LLM client and policy index (transcribing scanned sources once)."""
        config = load_config()
        client = LLMClient(settings, cache_only=cache_only)
        index = await PolicyIndex.aload(settings, client, config)
        jev = (
            JevClient(settings, cache_only=cache_only) if settings.jev_enabled else None
        )
        return cls(settings, config, client, index, jev)

    async def assess(
        self, case_id: str, exclude: frozenset[str] = frozenset()
    ) -> CaseResult:
        """Run the pipeline on one case.

        `exclude` lists document names (file stems) to leave out, for ablations.
        """
        token = case_scope.set(case_id)  # keeps concurrent cases' LLM traces apart
        try:
            with tracing.span(
                f"assess {case_id}",
                inputs={"case_id": case_id, "exclude": sorted(exclude)},
                metadata={
                    "run": tracing.run_name.get(),
                    "cheap_model": self.settings.cheap_model,
                    "strong_model": self.settings.strong_model,
                },
            ) as root:
                decision, bundle = await self._pipeline(case_id, exclude)
                result = CaseResult(
                    case_id,
                    decision,
                    bundle,
                    self.config,
                    self.client.trace_for(case_id),
                    self.jev.trace_for(case_id) if self.jev else [],
                )
                if root is not None:
                    root.add_outputs(_summary(result))
            return result
        finally:
            case_scope.reset(token)

    async def _pipeline(
        self, case_id: str, exclude: frozenset[str]
    ) -> tuple[Decision, AssessmentBundle]:
        """ingest -> facts -> site -> routing -> assessment -> synthesis."""
        settings, client, jev = self.settings, self.client, self.jev
        started = time.monotonic()
        with tracing.span("ingest"):
            pairs = load_documents(settings, case_id)
            pairs = [(doc, pdf) for doc, pdf in pairs if doc.name not in exclude]
            docs = await describe_drawings(client, settings.strong_model, pairs)
            docs = await classify_documents(client, settings.cheap_model, docs, jev)
            injection = await semantic_scan_injection(
                client, settings.cheap_model, docs, jev
            )
        with tracing.span("facts"):
            facts = await extract_facts(client, settings.cheap_model, docs)
        with tracing.span("site"):
            site = check_site(facts, docs)
        with tracing.span("routing") as span:
            routed = await route_issues(
                client, settings.cheap_model, facts, site, self.config, jev
            )
            issues = [r.issue for r in routed]
            if span is not None:
                span.add_outputs(
                    {
                        "issues": [i.value for i in issues],
                        "sources": {r.issue.value: r.source for r in routed},
                        "constraints_note": site.note,
                    }
                )
        logger.info(f"{case_id}: issues {[i.value for i in issues]}")
        with tracing.span("queries"):
            queries = await expand_queries(
                client, settings.cheap_model, facts, issues, self.config, site.note
            )
        with tracing.span("assess_issues"):
            assessments, citations = await assess_issues(
                client,
                settings.strong_model,
                self.index,
                self.config,
                issues,
                CaseContext(facts, docs, site, queries),
                settings.cheap_model,
                jev,
            )
        bundle = AssessmentBundle(
            case_id=case_id,
            facts=facts,
            constraints=site.constraints,
            constraints_note=site.note,
            assessments=assessments,
            citations=citations,
            routing=routed,
            review=build_review(assessments, facts, docs, site, self.config, injection),
            queries={i.value: qs for i, qs in queries.items()},
        )
        if jev is not None and jev.active("assess"):
            with tracing.span("jev_crosscheck"):
                bundle.review += await jev_crosscheck(jev, assessments, self.config)
        with tracing.span("verify"):
            bundle.review += await verify_decisive(
                client,
                settings.cheap_model,
                assessments,
                settings.verifier2_model,
                jev,
            )
        rejected, notes = rejected_conflicts(assessments, bundle.review)
        bundle.review += notes
        outcome = decide_outcome(assessments, require_grounding=True, rejected=rejected)
        if jev is not None:
            outcome, bundle.jev_decision, items = await jev_outcome(
                jev, assessments, self.config, outcome, settings.jev_decide
            )
            bundle.review += items
        with tracing.span("synthesis"):
            decision = await synthesise(client, settings.strong_model, bundle, outcome)
        bundle.review += reason_review(decision.reasons, bundle.assessments)
        if bundle.needs_officer:
            logger.warning(f"{case_id}: refer to officer ({_blocking(bundle)})")
        logger.info(
            f"{case_id}: {decision.decision.value} in {time.monotonic() - started:.0f}s"
        )
        return decision, bundle


def _blocking(bundle: AssessmentBundle) -> list[str]:
    """Kinds of the review items that put the case to the officer."""
    return sorted({item.kind for item in bundle.review if item.blocking})


def _summary(result: CaseResult) -> dict[str, object]:
    """Headline numbers attached to the case's tracing span."""
    return {
        "decision": result.decision.decision.value,
        "refer_to_officer": result.bundle.needs_officer,
        "n_reasons": len(result.decision.reasons),
        "n_conditions": len(result.decision.conditions),
        "issues": [a.issue.value for a in result.bundle.assessments],
        "n_citations": len(result.bundle.citations),
        "cost_usd": round(result.cost_usd, 4),
    }
