"""Typed artefacts passed between pipeline stages."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from domain import Finding, Issue
from tools.geospatial import LocationConstraints


class Consultee(BaseModel):
    """A consultation reply or public representation."""

    source: str
    position: str = Field(description="support, no objection, objection, or comments")
    summary: str
    requested_conditions: list[str] = Field(default_factory=list)


class CaseFacts(BaseModel):
    """Structured understanding of an application pack."""

    site_address: str
    postcode: str | None = None
    proposal: str = Field(description="One-sentence description of the proposal")
    development_type: str = Field(
        description="e.g. new dwelling, extension, change of use, conversion, HMO"
    )
    existing_use: str | None = None
    proposed_use: str | None = None
    dwelling_count: int | None = None
    key_measurements: list[str] = Field(
        default_factory=list,
        description="Dimensions, floor areas, storeys, parking spaces, distances",
    )
    consultees: list[Consultee] = Field(default_factory=list)
    flags: list[str] = Field(
        default_factory=list,
        description="Constraints or concerns mentioned: flood, trees, heritage, "
        "contamination, noise, HMO, gas pipeline, objections etc.",
    )
    missing_information: list[str] = Field(default_factory=list)


class Confidence(StrEnum):
    """How sure the assessor is; anything but a valid value is read as low."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Claim(BaseModel):
    """One fact the assessment relies on, with the exact words that support it.

    `grounded` is never set by the model: `pipeline.grounding` fills it in after
    checking the quote against the policy extracts or the application pack.
    """

    statement: str = Field(description="The fact or policy requirement relied on")
    source: Literal["policy", "application"]
    document: str = Field(
        default="",
        description="Application document name, or the policy extract's source label",
    )
    quote: str = Field(
        description="Verbatim words (copied exactly) from that document or extract"
    )
    grounded: bool | None = None


class IssueAssessment(BaseModel):
    """LLM assessment of one planning issue against retrieved policy."""

    issue: Issue
    finding: Finding
    substantive: bool = Field(
        default=False,
        description="True only if a conflict is serious enough to justify refusal",
    )
    rationale: str
    claims: list[Claim] = Field(
        default_factory=list,
        description="The facts and policy requirements the finding rests on, each "
        "with a verbatim quote",
    )
    evidence: list[str] = Field(
        default_factory=list, description="Document names/quotes relied on"
    )
    policy_refs: list[str] = Field(
        default_factory=list,
        description="e.g. 'Doncaster Local Plan Policy 44', 'NPPF paragraph 135'",
    )
    unsupported_refs: list[str] = Field(
        default_factory=list,
        description="Cited references that were not in the extracts (set by code)",
    )
    proposed_conditions: list[str] = Field(default_factory=list)
    failed: bool = Field(
        default=False,
        description="True if no model reply could be obtained (set by code)",
    )
    confidence: Confidence = Field(
        default=Confidence.MEDIUM, description="low, medium or high"
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _lenient_confidence(cls, value: object) -> object:
        """Read an unrecognised confidence as low, so it is checked rather than trusted."""
        text = str(value).strip().lower()
        return text if text in {c.value for c in Confidence} else Confidence.LOW

    @property
    def grounded_claims(self) -> list[Claim]:
        """Claims whose quote was found in the cited text."""
        return [c for c in self.claims if c.grounded]


ReviewKind = Literal[
    "ungrounded_conflict",
    "unsupported_reference",
    "missing_evidence",
    "unclear_core_issue",
    "inconsistent_documents",
    "verifier_disagreement",
    "ungrounded_reason",
    "injection_suspected",
    "truncated_document",
    "site_unknown",
    "assessment_failed",
    "jev_crosscheck",
    "jev_override",
]


class ReviewItem(BaseModel):
    """Something an officer must look at; `blocking` items put the case in 'refer'."""

    kind: ReviewKind
    detail: str
    issue: Issue | None = None
    blocking: bool = False
    subject: str = Field(default="", description="e.g. the document that is missing")


class IssueChoice(BaseModel):
    """One issue the routing model wants assessed, with its reason."""

    issue: Issue
    reason: str = Field(description="One sentence naming the fact that calls for it")


class RoutingChoice(BaseModel):
    """What the routing model returns: extra issues beyond the always-assessed ones."""

    issues: list[IssueChoice] = Field(default_factory=list)


class RoutedIssue(BaseModel):
    """An issue chosen for assessment and why (kept for the audit brief)."""

    issue: Issue
    source: Literal["core", "constraint", "jev", "llm", "keyword"]
    reason: str = ""


class JevDecisionNote(BaseModel):
    """Jev's independent read of the final outcome, kept so any override can be audited."""

    mode: Literal["shadow", "on"]
    rule: Literal["approve", "refuse"] = Field(description="Outcome of decide_outcome")
    jev: Literal["approve", "refuse"] = Field(description="Jev's more probable outcome")
    p_refuse: float
    confidence: float = Field(description="|2p - 1|: distance from a coin flip")
    tau: float
    model: str
    overridden: bool = Field(
        description="True if Jev's outcome replaced the rule's (mode on, confident, differs)"
    )


class AssessmentBundle(BaseModel):
    """Everything the synthesis stage and the audit brief need."""

    case_id: str
    facts: CaseFacts
    constraints: LocationConstraints | None
    constraints_note: str
    assessments: list[IssueAssessment]
    citations: dict[str, list[str]] = Field(
        default_factory=dict, description="issue -> retrieved passage citations"
    )
    routing: list[RoutedIssue] = Field(
        default_factory=list, description="why each issue was assessed"
    )
    review: list[ReviewItem] = Field(
        default_factory=list, description="items an officer must check"
    )
    queries: dict[str, list[str]] = Field(
        default_factory=dict,
        description="issue -> extra search queries written for the case",
    )
    jev_decision: JevDecisionNote | None = Field(
        default=None,
        description="Jev's check of the outcome, when JEV_DECIDE is not off",
    )

    @property
    def needs_officer(self) -> bool:
        """True when the evidence does not safely support a draft outcome."""
        return any(item.blocking for item in self.review)
