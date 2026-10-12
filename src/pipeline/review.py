"""Collect everything that stops the draft from being safely relied on.

Deterministic rules turn the pipeline's artefacts into `ReviewItem`s. A blocking item
means the evidence does not support a firm draft, so the case is put to the officer
('refer') instead of being approved by default.
"""

from domain import Finding, Issue
from pipeline.assess import PACK_CHARS
from pipeline.consistency import consistency_review
from pipeline.doctype import kinds_of
from pipeline.facts import MAX_DOC_CHARS
from pipeline.guard import InjectionFlag, render_pack, scan_injection
from pipeline.ingest import CaseDocument
from pipeline.models import CaseFacts, IssueAssessment, ReviewItem
from pipeline.site import SiteReport
from policy.config import EvidenceRequirement, PolicyConfig
from tools.geospatial import LocationConstraints


def _applies(
    requirement: EvidenceRequirement, site: LocationConstraints | None
) -> bool:
    if requirement.applies_if is None:
        return True
    if site is None:
        return False
    if requirement.applies_if == "flood_zone":
        return bool(site.flood_risk_level)
    return bool(site.conservation_area_name or site.listed_building_grade)


def missing_evidence(
    issue: Issue,
    config: PolicyConfig,
    docs: list[CaseDocument],
    site: LocationConstraints | None,
) -> list[str]:
    """Labels of the issue's required documents that no document is classified as."""
    present = set().union(*(kinds_of(d) for d in docs)) if docs else set()
    missing = []
    for requirement in config.recipes[issue].required_evidence:
        if not _applies(requirement, site):
            continue
        if not present.intersection(requirement.kinds):
            missing.append(requirement.label)
    return missing


def assessment_review(
    assessment: IssueAssessment,
    config: PolicyConfig,
    docs: list[CaseDocument],
    site: LocationConstraints | None,
) -> list[ReviewItem]:
    """Review items for one assessed issue."""
    issue = assessment.issue
    items: list[ReviewItem] = []
    if assessment.failed:
        return [
            ReviewItem(
                kind="assessment_failed",
                issue=issue,
                blocking=True,
                detail=assessment.rationale,
            )
        ]
    decisive = assessment.finding == Finding.CONFLICTS and assessment.substantive
    if decisive and not assessment.grounded_claims:
        items.append(
            ReviewItem(
                kind="ungrounded_conflict",
                issue=issue,
                blocking=True,
                detail="A refusable conflict is asserted but none of its quotes could "
                "be found in the policy extracts or application documents.",
            )
        )
    if assessment.unsupported_refs:
        items.append(
            ReviewItem(
                kind="unsupported_reference",
                issue=issue,
                blocking=decisive,
                detail="Cited but not among the extracts shown: "
                + "; ".join(assessment.unsupported_refs),
            )
        )
    if assessment.finding == Finding.UNCLEAR:
        core = issue in config.routing.core
        items.append(
            ReviewItem(
                kind="unclear_core_issue" if core else "missing_evidence",
                issue=issue,
                blocking=core,
                detail=assessment.rationale,
            )
        )
    for label in missing_evidence(issue, config, docs, site):
        items.append(
            ReviewItem(
                kind="missing_evidence",
                issue=issue,
                blocking=True,
                subject=label,
                detail=f"Pack has no {label}.",
            )
        )
    return items


def build_review(
    assessments: list[IssueAssessment],
    facts: CaseFacts,
    docs: list[CaseDocument],
    site: SiteReport,
    config: PolicyConfig,
    model_flags: list[InjectionFlag] | None = None,
) -> list[ReviewItem]:
    """All review items for a case, in a stable order.

    `model_flags` are the passages the model-based injection pass found, added to the
    regex flags (an excerpt both found is listed once).
    """
    items: list[ReviewItem] = []
    for assessment in assessments:
        items += assessment_review(assessment, config, docs, site.constraints)
    items += consistency_review(facts, docs)
    if site.constraints is None:
        items.append(ReviewItem(kind="site_unknown", detail=site.note))
    flags = scan_injection(docs)
    seen = {(f.document, f.excerpt) for f in flags}
    flags += [f for f in model_flags or [] if (f.document, f.excerpt) not in seen]
    for flag in flags:
        items.append(
            ReviewItem(
                kind="injection_suspected",
                detail=f"{flag.document}: {flag.kind}: “{flag.excerpt}”",
            )
        )
    for stage, limit in (("facts", MAX_DOC_CHARS), ("assessment", PACK_CHARS)):
        for notice in render_pack(docs, limit, stage).truncated:
            items.append(
                ReviewItem(
                    kind="truncated_document",
                    detail=f"{notice.document}: only {notice.shown_chars} of "
                    f"{notice.total_chars} characters were shown at the {stage} stage.",
                )
            )
    return items
