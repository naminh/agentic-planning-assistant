"""Officer-facing markdown brief: outcome, issue table, evidence and caveats."""

from decision import Decision
from domain import Finding
from pipeline.models import AssessmentBundle, IssueAssessment
from policy.config import PolicyConfig


def _routing_section(bundle: AssessmentBundle) -> list[str]:
    """Why each issue was assessed (omitted for runs saved before routing was recorded)."""
    if not bundle.routing:
        return []
    lines = [
        "## Why these issues were assessed",
        "",
        "| Issue | Chosen by | Reason |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r.issue.value} | {r.source} | {r.reason or '-'} |" for r in bundle.routing
    ]
    return [*lines, ""]


def _queries_section(bundle: AssessmentBundle) -> list[str]:
    """Extra search queries written for this case (audit trail for retrieval)."""
    if not bundle.queries:
        return []
    lines = ["## Extra policy search queries", ""]
    lines += [f"- {issue}: {'; '.join(qs)}" for issue, qs in bundle.queries.items()]
    return [*lines, ""]


def _referral_section(bundle: AssessmentBundle) -> list[str]:
    """Banner and table of review items; the banner appears only for blocking ones."""
    if not bundle.review:
        return []
    lines: list[str] = []
    if bundle.needs_officer:
        lines += [
            "> **REFER TO OFFICER.** The evidence does not safely support this draft "
            "on its own. The recommendation below is a provisional default; resolve "
            "the blocking items first.",
            "",
        ]
    lines += [
        "## Review items",
        "",
        "| Blocking | Kind | Issue | Detail |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {'yes' if i.blocking else 'no'} | {i.kind} | "
        f"{i.issue.value if i.issue else '-'} | {' '.join(i.detail.split())} |"
        for i in bundle.review
    ]
    return [*lines, ""]


def _jev_section(bundle: AssessmentBundle) -> list[str]:
    """Jev's independent read of the outcome and whether it changed the draft."""
    note = bundle.jev_decision
    if note is None:
        return []
    lines = [
        "## Jev outcome check",
        "",
        f"- Mode: {note.mode}; model: {note.model}",
        f"- Rule outcome: {note.rule}; Jev outcome: {note.jev} "
        f"(p(refuse) {note.p_refuse}, confidence {note.confidence}, "
        f"threshold {note.tau})",
    ]
    if note.overridden:
        lines.append(
            f"- **OVERRIDE:** the draft follows Jev ({note.jev}) instead of the "
            f"rule ({note.rule})."
        )
    elif note.jev != note.rule:
        lines.append(
            "- Jev disagrees but the draft keeps the rule outcome "
            + ("(shadow mode)." if note.mode == "shadow" else "(below threshold).")
        )
    return [*lines, ""]


def _claims_lines(a: IssueAssessment) -> list[str]:
    """Each claim with a mark showing whether its quote was found in the source."""
    marks = {True: "verified", False: "NOT FOUND", None: "unchecked"}
    return [
        f"- [{marks[c.grounded]}] {c.statement} ({c.source}: {c.document or 'n/a'}) "
        f"“{' '.join(c.quote.split())}”"
        for c in a.claims
    ]


def audit_brief(
    bundle: AssessmentBundle, decision: Decision, config: PolicyConfig
) -> str:
    """Officer-facing markdown: outcome, issue table, evidence, caveats."""
    lines = [
        f"# Draft officer brief: {bundle.case_id}",
        "",
        f"**Recommendation:** {decision.decision.value.upper()} "
        "(draft for officer review, not a decision)",
        "",
        *_referral_section(bundle),
        f"**Proposal:** {bundle.facts.proposal}",
        f"**Site:** {bundle.facts.site_address}",
        f"**Constraints:** {bundle.constraints_note}",
        "",
        "## Issue assessments",
        "",
        "| Issue | Finding | Confidence | Policy basis |",
        "|---|---|---|---|",
    ]
    for a in bundle.assessments:
        mark = f"{a.finding.value}{' (substantive)' if a.substantive else ''}"
        lines.append(
            f"| {a.issue.value} | {mark} | {a.confidence} | {'; '.join(a.policy_refs) or '-'} |"
        )
    lines += ["", "## Rationale and evidence", ""]
    for a in bundle.assessments:
        lines.append(f"### {a.issue.value}")
        lines.append(a.rationale)
        if a.claims:
            lines.append("")
            lines += _claims_lines(a)
        if a.unsupported_refs:
            lines += [
                "",
                f"- Unsupported references removed: {'; '.join(a.unsupported_refs)}",
            ]
        if a.evidence:
            lines.append("")
            lines += [f"- Evidence: {e}" for e in a.evidence]
        cites = bundle.citations.get(a.issue.value, [])
        if cites:
            lines.append(f"- Policy passages consulted: {', '.join(cites[:8])}")
        lines.append("")
    lines += _jev_section(bundle)
    lines += _routing_section(bundle)
    lines += _queries_section(bundle)
    needs_check = [
        a
        for a in bundle.assessments
        if a.confidence == "low" or a.finding == Finding.UNCLEAR
    ]
    fallback = any(r.source == "keyword" for r in bundle.routing)
    lines += ["## Officer should check", ""]
    if fallback:
        lines.append(
            "- Routing: the routing model failed, so issues were chosen by keyword "
            "matching; an issue the keywords miss was not assessed."
        )
    if needs_check:
        lines += [f"- {a.issue.value}: {a.rationale}" for a in needs_check]
    if bundle.facts.missing_information:
        lines += [f"- Missing: {m}" for m in bundle.facts.missing_information]
    if not needs_check and not bundle.facts.missing_information and not fallback:
        lines.append("- No low-confidence items flagged.")
    versions = config.versions_for(
        c for cites in bundle.citations.values() for c in cites
    )
    if versions:
        lines += ["", "## Policy sources consulted (edition)", ""]
        lines += [f"- {label}: {version}" for label, version in versions.items()]
    lines += ["", decision.to_markdown()]
    return "\n".join(lines)
