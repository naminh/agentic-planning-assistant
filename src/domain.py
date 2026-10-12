"""Core vocabulary shared by `pipeline` and `policy` (a leaf module: imports nothing local)."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

# Source labels the code treats specially (must match `label` in sources.yaml).
LOCAL_PLAN = "Doncaster Local Plan"
NPPF = "NPPF"


# `configs/<council>/issues.yaml` must use exactly these ids; adding a new issue
# category is a one-line change here plus a recipe in that file.
class Issue(StrEnum):
    """Planning issues an officer would address, mirroring the reference decisions."""

    PRINCIPLE = "principle_of_development"
    DESIGN = "design_and_character"
    HERITAGE = "heritage_and_conservation"
    AMENITY = "residential_amenity"
    HIGHWAYS = "highways_and_parking"
    FLOOD = "flood_risk_and_drainage"
    TREES_ECOLOGY = "trees_and_ecology"
    CONTAMINATION = "contamination_and_pollution"
    HMO = "houses_in_multiple_occupation"
    RURAL = "countryside_and_green_belt"
    PUBLIC_HEALTH = "public_health_and_hot_food"
    COMMUNITY = "community_facilities"


class Finding(StrEnum):
    """Outcome of assessing a single issue."""

    COMPLIES = "complies"
    CONFLICTS = "conflicts"
    UNCLEAR = "unclear"


ChunkKind = Literal["policy", "supporting", "table", "figure", "text"]


class PolicyChunk(BaseModel):
    """A retrievable passage with enough metadata to cite it.

    A chunk is one part of a logical unit (a policy's clauses, its supporting text, a
    table, a figure): `unit_id` is shared by all parts, `part`/`parts` say where this
    one sits, and continuation parts repeat the unit's heading so each stands alone.
    """

    source: str
    page: int  # page the chunk starts on
    policy_id: str | None = None
    text: str
    version: str | None = None  # edition of the source; never shown to the model
    kind: ChunkKind = "text"
    heading: str | None = None  # nearest heading, or a table/figure caption
    unit_id: str | None = None
    part: int = 1
    parts: int = 1
    end_page: int | None = None  # page the chunk ends on, if it spans pages
    image: str | None = None  # page-image path for figure chunks

    @property
    def citation(self) -> str:
        """Human-readable reference used in the audit brief."""
        policy = f" ({self.policy_id})" if self.policy_id else ""
        return f"{self.source} p.{self.page}{policy}"


DocKind = Literal[
    "drawings",
    "heritage_statement",
    "design_and_access_statement",
    "flood_risk_assessment",
    "application_form",
    "consultation_reply",
    "supporting_statement",
    "other",
]
