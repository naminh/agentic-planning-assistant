"""Turn issue assessments into the final Decision."""

import json
import re
from collections.abc import Collection

from loguru import logger

from config import JevDecide
from decision import Decision, DecisionType
from domain import Finding, Issue
from jev import JevClient, JevError, noul
from llm import LLMClient
from pipeline.crosscheck import crosscheck_state
from pipeline.models import (
    AssessmentBundle,
    IssueAssessment,
    JevDecisionNote,
    ReviewItem,
)
from pipeline.verify import rejected_conflicts
from policy.config import PolicyConfig

STANDARD_CONDITIONS = [
    "The development hereby permitted shall be begun before the expiration of three "
    "years from the date of this permission. Reason: Required by Section 91 of the "
    "Town and Country Planning Act 1990.",
    "The development hereby permitted shall be carried out in accordance with the "
    "approved plans and documents listed on the decision notice. Reason: To ensure "
    "that the development is carried out in accordance with the application as "
    "approved.",
]

SYNTH_SYSTEM = """\
You write the final officer decision for a Doncaster housing planning application \
from issue assessments that have already been made. The outcome is fixed: {outcome}.

Reasons:
- One self-contained reason per assessed issue (including issues that are acceptable), \
{refusal_order}
- Each reason states the finding and why, then the policy basis (e.g. 'in accordance \
with Doncaster Local Plan Policy 44 and NPPF paragraph 135' or 'contrary to ...'). \
Use only the policy references present in the assessments. No policy-only reasons.
- If two assessments rest on the same ground (e.g. principle and public health), merge
  them into one reason rather than repeating it.
- Neutral planning language ('is considered acceptable', 'would harm').
- Do not mention internal notes or this tool; do not invent facts.

Conditions:
{condition_rules}"""

REFUSE_RULES = "Return an empty conditions list."
APPROVE_RULES = (
    "Merge and de-duplicate the proposed conditions from the assessments into a "
    "concise list (aim for 3-12), each as full condition text followed by "
    "'Reason: ...'. Do not include the time-limit or approved-plans conditions "
    "(they are added automatically). Do not add informatives."
)


def decide_outcome(
    assessments: list[IssueAssessment],
    *,
    require_grounding: bool = False,
    rejected: Collection[Issue] = (),
) -> DecisionType:
    """Refuse only on substantive, evidenced conflicts; otherwise approve.

    With `require_grounding`, a conflict counts only if at least one of its quotes was
    found in the text it cites. An unevidenced conflict is not a basis for refusal; the
    case is referred to the officer instead (see `pipeline.review`). Issues in `rejected`
    (conflicts the verifiers turned down, see `pipeline.verify.rejected_conflicts`)
    do not count.
    """
    for a in assessments:
        if a.issue in rejected:
            continue
        if (
            a.finding == Finding.CONFLICTS
            and a.substantive
            and (not require_grounding or a.grounded_claims)
        ):
            return DecisionType.REFUSE
    return DecisionType.APPROVE


JEV_STATE_CHARS = 60_000  # Jev reads up to 32k tokens


async def jev_outcome(
    jev: JevClient,
    assessments: list[IssueAssessment],
    config: PolicyConfig,
    rule: DecisionType,
    mode: JevDecide,
) -> tuple[DecisionType, JevDecisionNote | None, list[ReviewItem]]:
    """Jev's own read of whether the verified quotes justify refusal, set beside the rule.

    `decide_outcome` stays the baseline. Jev is shown only each issue's grounded quotes
    (no findings, rationale or consultee views) and asked one yes/no question. If it
    differs from the rule with confidence at or above the `decide` threshold and `mode`
    is "on", Jev's outcome is used and a blocking review item records the override; in
    "shadow" mode (or below the threshold, or if Jev fails) the rule's outcome stands.

    Returns:
        The outcome to use, the note for the bundle (None if Jev could not answer),
        and review items (one blocking item per override).
    """
    if mode == "off":
        return rule, None, []
    sections = [
        crosscheck_state(config.recipes[a.issue].label, a.grounded_claims)
        for a in assessments
        if a.grounded_claims and not a.failed
    ]
    try:
        result = await jev.decide(
            "\n\n=====\n\n".join(sections)[:JEV_STATE_CHARS] or "(no verified quotes)",
            {
                "refuse": noul(
                    "Taking every issue above together, is there at least one issue "
                    "where the quoted application facts conflict with the quoted "
                    "policy wording so seriously that the application should be "
                    "refused, because no planning condition could cure the conflict?",
                    yes="At least one issue shows a conflict that conditions cannot "
                    "cure, so the application should be refused.",
                    no="No issue shows such a conflict, so the application should "
                    "be approved, with conditions where needed.",
                )
            },
            stage="decide",
        )
    except JevError as err:
        logger.warning(f"Jev outcome check unavailable ({err}); using the rule")
        return rule, None, []
    answer = result.noul("refuse")
    vote = DecisionType.REFUSE if answer.p >= 0.5 else DecisionType.APPROVE
    tau = jev.tau("decide")
    overridden = mode == "on" and vote != rule and answer.confidence >= tau
    note = JevDecisionNote(
        mode=mode,
        rule=rule.value,
        jev=vote.value,
        p_refuse=round(answer.p, 3),
        confidence=round(answer.confidence, 3),
        tau=tau,
        model=result.model,
        overridden=overridden,
    )
    if not overridden:
        return rule, note, []
    logger.warning(
        f"Jev overrides the rule: {rule.value} -> {vote.value} "
        f"(p_refuse {answer.p:.2f}, confidence {answer.confidence:.2f} >= {tau})"
    )
    item = ReviewItem(
        kind="jev_override",
        blocking=True,
        detail=f"Jev ({result.model}) overruled the rule outcome '{rule.value}' with "
        f"'{vote.value}' (p_refuse {answer.p:.2f}, confidence {answer.confidence:.2f}, "
        f"threshold {tau}). Check the verified quotes before relying on it.",
    )
    return vote, note, [item]


def evidence_conditions(review: list[ReviewItem]) -> list[str]:
    """Conditions that secure documents missing from the pack, for an approval draft."""
    return [
        f"No development shall commence until {item.subject} has been submitted to and "
        "approved in writing by the Local Planning Authority. Reason: The application "
        "did not include this information, which is needed to assess the proposal."
        for item in review
        if item.kind == "missing_evidence" and item.blocking and item.subject
    ]


_REASON_POLICY = re.compile(r"\bpolic(?:y|ies)\s+(\d{1,3})\b", re.I)
_REASON_PARAGRAPH = re.compile(r"\bparagraphs?\s+(\d{1,3})\b", re.I)


def reason_review(
    reasons: list[str], assessments: list[IssueAssessment]
) -> list[ReviewItem]:
    """Flag reasons that cite a policy number no assessment cited.

    The synthesis model sees only the assessments, so every reference in a reason must
    come from one of them; anything else was invented while writing up.
    """
    cited = " ".join(ref for a in assessments for ref in a.policy_refs)
    known_policies = set(_REASON_POLICY.findall(cited))
    known_paragraphs = set(_REASON_PARAGRAPH.findall(cited))
    items = []
    for reason in reasons:
        stray = [
            f"Policy {n}"
            for n in _REASON_POLICY.findall(reason)
            if n not in known_policies
        ] + [
            f"paragraph {n}"
            for n in _REASON_PARAGRAPH.findall(reason)
            if n not in known_paragraphs
        ]
        if stray:
            items.append(
                ReviewItem(
                    kind="ungrounded_reason",
                    blocking=True,
                    detail=f"Reason cites {', '.join(stray)}, which no assessment "
                    f"supported: {reason[:120]}",
                )
            )
    return items


async def synthesise(
    client: LLMClient,
    model: str,
    bundle: AssessmentBundle,
    outcome: DecisionType | None = None,
) -> Decision:
    """Draft reasons/conditions for `outcome`, by default the rule-determined one."""
    if outcome is None:
        rejected, _ = rejected_conflicts(bundle.assessments, bundle.review)
        outcome = decide_outcome(
            bundle.assessments, require_grounding=True, rejected=rejected
        )
    refuse = outcome == DecisionType.REFUSE
    system = SYNTH_SYSTEM.format(
        outcome=outcome.value,
        refusal_order=(
            "refusal grounds first, then the acceptable aspects."
            if refuse
            else "most important issues first."
        ),
        condition_rules=REFUSE_RULES if refuse else APPROVE_RULES,
    )
    payload = [a.model_dump(mode="json") for a in bundle.assessments]
    drafted = await client.parse(
        Decision,
        model,
        system,
        f"Proposal: {bundle.facts.proposal}\n\nASSESSMENTS:\n{json.dumps(payload, indent=1)}",
        max_tokens=8000,
        stage="synthesis",
    )
    conditions = (
        []
        if refuse
        else [
            *STANDARD_CONDITIONS,
            *drafted.conditions,
            *evidence_conditions(bundle.review),
        ]
    )
    return Decision(decision=outcome, reasons=drafted.reasons, conditions=conditions)
