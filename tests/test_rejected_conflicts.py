"""A refusal every verifier turned down, or that rests on the same evidence, does not decide."""

from decision import DecisionType
from domain import Finding, Issue
from pipeline.decide import decide_outcome
from pipeline.models import Claim, IssueAssessment, ReviewItem
from pipeline.verify import rejected_conflicts


def conflict(issue: Issue, *quotes: str) -> IssueAssessment:
    """Each quote "doc:text" comes from application document `doc` (default highways)."""
    claims = [
        Claim(
            statement="s",
            source="application",
            document=q.split(":")[0] if ":" in q else "highways",
            quote=q,
            grounded=True,
        )
        for q in quotes
    ]
    claims.append(
        Claim(
            statement="p",
            source="policy",
            document="plan",
            quote=f"policy for {issue.value}",
            grounded=True,
        )
    )
    return IssueAssessment(
        issue=issue,
        finding=Finding.CONFLICTS,
        substantive=True,
        rationale="r",
        claims=claims,
    )


def rejection(issue: Issue, blocking: bool = True) -> ReviewItem:
    return ReviewItem(
        kind="verifier_disagreement", issue=issue, blocking=blocking, detail="no"
    )


def refuses(assessments: list[IssueAssessment], review: list[ReviewItem]) -> bool:
    rejected, _ = rejected_conflicts(assessments, review)
    outcome = decide_outcome(assessments, require_grounding=True, rejected=rejected)
    return outcome == DecisionType.REFUSE


def test_a_conflict_every_verifier_rejected_does_not_refuse() -> None:
    only = [conflict(Issue.HIGHWAYS, "cannot turn")]
    assert refuses(only, [])
    assert not refuses(only, [rejection(Issue.HIGHWAYS)])


def test_a_second_verifier_that_supported_it_keeps_the_refusal() -> None:
    only = [conflict(Issue.HIGHWAYS, "cannot turn")]
    assert refuses(only, [rejection(Issue.HIGHWAYS, blocking=False)])


def test_case_004_shape_shared_evidence_is_rejected_too() -> None:
    highways = conflict(Issue.HIGHWAYS, "cannot turn")
    design = conflict(Issue.DESIGN, "cannot turn")  # same quote, other issue
    review = [rejection(Issue.HIGHWAYS)]
    rejected, notes = rejected_conflicts([highways, design], review)
    assert rejected == {Issue.HIGHWAYS, Issue.DESIGN}
    assert [n.issue for n in notes] == [Issue.DESIGN] and not notes[0].blocking
    assert not refuses([highways, design], review)


def test_case_001_shape_independent_evidence_still_refuses() -> None:
    design = conflict(Issue.DESIGN, "plans:overbearing two storey extension")
    amenity = conflict(Issue.AMENITY, "report:loss of light to neighbour")
    assert refuses([design, amenity], [])  # verifiers supported both
    # one rejected conflict does not cancel an independent one
    assert refuses([design, amenity], [rejection(Issue.AMENITY)])


def test_conflict_that_contains_the_rejected_defect_is_rejected_too() -> None:
    # case-004 as it really was: design also cites a supportive quote about the barn form
    highways = conflict(Issue.HIGHWAYS, "cannot turn")
    design = conflict(
        Issue.DESIGN, "statement:traditional barn form", "cannot turn again"
    )
    assert not refuses([highways, design], [rejection(Issue.HIGHWAYS)])


def test_unrelated_evidence_is_not_propagated() -> None:
    highways = conflict(Issue.HIGHWAYS, "cannot turn")
    design = conflict(
        Issue.DESIGN, "statement:poor materials", "statement:dominant roof"
    )
    assert refuses([highways, design], [rejection(Issue.HIGHWAYS)])
