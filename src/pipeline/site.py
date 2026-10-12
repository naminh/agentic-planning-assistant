"""Check the site against the geospatial dataset and describe what it says."""

from dataclasses import dataclass

from pipeline.consistency import POSTCODE_RE, postcode_conflict
from pipeline.ingest import CaseDocument
from pipeline.models import CaseFacts
from tools.geospatial import LocationConstraints, postcode_lookup


@dataclass(frozen=True)
class SiteReport:
    """Constraints at the site, or None with a note saying why they are unknown."""

    constraints: LocationConstraints | None
    note: str


def find_postcode(facts: CaseFacts, docs: list[CaseDocument]) -> str | None:
    """Prefer the model's postcode; fall back to the first one in the text."""
    if facts.postcode and POSTCODE_RE.search(facts.postcode.upper()):
        return str(facts.postcode.upper())
    for doc in docs:
        match = POSTCODE_RE.search(doc.text)
        if match:
            return match.group(0)
    return None


def check_site(facts: CaseFacts, docs: list[CaseDocument]) -> SiteReport:
    """Look up the case's postcode; unknown sites are reported, never "unconstrained".

    If the extracted postcode contradicts the application form the site is treated as
    unknown rather than trusting either.
    """
    conflict = postcode_conflict(facts, docs)
    if conflict:
        return SiteReport(
            None,
            f"{conflict} Flood, heritage and green belt status are UNKNOWN (not 'none').",
        )
    return lookup_constraints(find_postcode(facts, docs))


def lookup_constraints(postcode: str | None) -> SiteReport:
    """Run the hardcoded geospatial lookup and explain unknown results."""
    if not postcode:
        return SiteReport(
            None, "No postcode found: site constraints could not be checked."
        )
    result = postcode_lookup(postcode)
    if not result["found"]:
        return SiteReport(
            None,
            f"Postcode {postcode} is not in the constraint dataset: flood, heritage "
            "and green belt status are UNKNOWN (not 'none').",
        )
    constraints = LocationConstraints(**result["constraints"])  # type: ignore[arg-type]
    return SiteReport(constraints, _describe(postcode, constraints))


def _describe(postcode: str, constraints: LocationConstraints) -> str:
    flood = (
        f"Flood Zone(s) {constraints.flood_risk_level}"
        if constraints.flood_risk_level
        else "not in Flood Zone 2 or 3 (treat as Flood Zone 1)"
    )
    other = [
        f"{label}: {value}"
        for label, value in (
            ("conservation area", constraints.conservation_area_name),
            ("listed building grade", constraints.listed_building_grade),
            ("heritage at risk", constraints.heritage_at_risk_name),
            ("green belt", "yes" if constraints.is_green_belt else None),
        )
        if value
    ]
    return "; ".join([f"Constraints looked up for {postcode}", flood, *other])
