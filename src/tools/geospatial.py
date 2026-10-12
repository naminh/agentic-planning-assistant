"""Geospatial tools which map postcodes in the dataset to geospatial constraints."""

from dataclasses import asdict, dataclass, field
from typing import Literal

FloodZone = Literal[2, 3]
ListedBuildingGrade = Literal["I", "II*", "II"]


@dataclass(frozen=True)
class LocationConstraints:
    """Geospatial constraints associated with a known postcode."""

    conservation_area_name: str | None = None
    flood_risk_level: list[FloodZone] = field(default_factory=list)
    is_green_belt: bool = False
    listed_building_grade: ListedBuildingGrade | None = None
    heritage_at_risk_name: str | None = None


def _normalize_postcode(postcode: str) -> str:
    """Normalize a postcode for hashmap lookup.

    Args:
        postcode: User-supplied postcode.

    Returns:
        Uppercase postcode with all whitespace removed.
    """
    return "".join(postcode.upper().split())


POSTCODE_LOCATION_LOOKUP: dict[str, LocationConstraints] = {
    "ZZ251WD": LocationConstraints(
        conservation_area_name="doncaster_high_street",
    ),
    "ZZ296EP": LocationConstraints(),
    "ZZ933JC": LocationConstraints(),
    "ZZ460BG": LocationConstraints(
        flood_risk_level=[2],
    ),
    "ZZ385BX": LocationConstraints(
        flood_risk_level=[3],
    ),
    "ZZ540FT": LocationConstraints(),
    "ZZ388VR": LocationConstraints(
        flood_risk_level=[2, 3],
    ),
    "ZZ170TB": LocationConstraints(),
    "ZZ637KS": LocationConstraints(
        conservation_area_name="bessacarr",
    ),
    "ZZ432JG": LocationConstraints(),
    "ZZ928CM": LocationConstraints(
        flood_risk_level=[2],
    ),
}


def postcode_lookup(postcode: str) -> dict[str, object]:
    """Return geospatial lookup data for a postcode.

    Args:
        postcode: User-supplied postcode to look up.

    Returns:
        A dictionary with the original postcode, whether it was found, and any
        matching geospatial constraints.
    """
    normalized_postcode = _normalize_postcode(postcode)
    planning_constraints = POSTCODE_LOCATION_LOOKUP.get(normalized_postcode)

    if planning_constraints is None:
        return {
            "found": False,
            "postcode": postcode.strip(),
            "constraints": None,
        }

    return {
        "found": True,
        "postcode": postcode.strip(),
        "constraints": asdict(planning_constraints),
    }
