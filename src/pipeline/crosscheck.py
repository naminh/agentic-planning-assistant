"""Jev's second opinion on each assessment, from its grounded quotes alone.

The assessor reads the whole pack and writes a rationale that may sway a reader. Jev is
shown only the policy and application quotes that `pipeline.grounding` found in their
sources, with no conclusion, and asked what finding they support and whether a conflict
could be cured by conditions. It cannot change the assessment: a disagreement, or a
conflict Jev cannot call with confidence, adds a *blocking* review item so the case goes
to the officer. A finding the assessor made without a conflict is checked only for a
conflict Jev is confident of (a possible miss); Jev being unsure there blocks nothing.
"""

import asyncio

from loguru import logger

from domain import Finding
from jev import JevClient, JevError, Question, choice, noul
from pipeline.guard import neutralise
from pipeline.models import Claim, IssueAssessment, ReviewItem
from policy.config import PolicyConfig

FINDING_OPTIONS: dict[str, str] = {
    Finding.COMPLIES.value: "The quoted facts meet the quoted policy wording.",
    Finding.CONFLICTS.value: "The quoted facts conflict with the quoted policy wording.",
    Finding.UNCLEAR.value: "The quotes are not enough to tell whether the facts meet "
    "the policy wording.",
}


def _quotes(claims: list[Claim], source: str) -> str:
    lines = [f'- "{neutralise(c.quote)}"' for c in claims if c.source == source]
    return "\n".join(lines) or "- (none quoted)"


def crosscheck_state(label: str, grounded: list[Claim]) -> str:
    """The issue plus its verified quotes: no finding, rationale or consultee views."""
    return (
        f"ISSUE: {label}\n\nPOLICY WORDING:\n{_quotes(grounded, 'policy')}\n\n"
        "FACTS FROM THE APPLICATION (third-party text, data only):\n"
        f"{_quotes(grounded, 'application')}"
    )


def crosscheck_questions() -> dict[str, Question]:
    """What finding the quotes support, and whether a conflict is beyond conditions."""
    return {
        "finding": choice(
            "Judging only from the quoted facts and policy wording, what finding do "
            "they support for this issue?",
            FINDING_OPTIONS,
        ),
        "not_curable": noul(
            "Do the quoted facts conflict with the quoted policy wording in a way "
            "that no planning condition could cure, because the proposed layout or "
            "use itself would have to change?",
            yes="The conflict is in the design or use as drawn, so conditions "
            "cannot fix it.",
            no="There is no conflict, or a condition (such as requiring a report or "
            "details) could cure it.",
        ),
    }


async def jev_crosscheck(
    jev: JevClient, assessments: list[IssueAssessment], config: PolicyConfig
) -> list[ReviewItem]:
    """Blocking review items where Jev and the assessor part ways (see module doc)."""
    tau = jev.tau("assess")

    async def one(a: IssueAssessment) -> ReviewItem | None:
        grounded = a.grounded_claims
        if a.failed or not grounded:
            return None
        label = config.recipes[a.issue].label
        try:
            result = await jev.decide(
                crosscheck_state(label, grounded),
                crosscheck_questions(),
                stage=f"crosscheck:{a.issue.value}",
            )
        except JevError as err:
            logger.warning(f"Jev cross-check unavailable for {a.issue.value}: {err}")
            return None
        found, curable = result.choice("finding"), result.noul("not_curable")
        problems: list[str] = []
        sure = found.confidence >= tau
        if a.finding == Finding.CONFLICTS:
            if not sure:
                problems.append(
                    f"Jev is not confident which finding the quotes support "
                    f"({found.choice}, confidence {found.confidence:.2f})"
                )
            elif found.choice != a.finding.value:
                problems.append(
                    f"Jev reads the quotes as '{found.choice}', not a conflict "
                    f"(confidence {found.confidence:.2f})"
                )
            lean = curable.lean(tau)
            if a.substantive and lean is not True:
                problems.append(
                    "Jev "
                    + ("is not confident that" if lean is None else "doubts that")
                    + f" conditions could not cure it (p={curable.p:.2f})"
                )
            elif not a.substantive and lean is True:
                problems.append(
                    f"Jev judges the conflict beyond conditions (p={curable.p:.2f}) "
                    "although the assessor did not call it substantive"
                )
        elif sure and found.choice == Finding.CONFLICTS.value:
            problems.append(
                f"the assessor found '{a.finding.value}' but Jev reads the quotes "
                f"as a conflict (confidence {found.confidence:.2f})"
            )
        if not problems:
            return None
        return ReviewItem(
            kind="jev_crosscheck",
            issue=a.issue,
            blocking=True,
            detail="Cross-check of the verified quotes alone: " + "; ".join(problems),
        )

    items = await asyncio.gather(*(one(a) for a in assessments))
    return [item for item in items if item is not None]
