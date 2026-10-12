# Geospatial tools

This package provides a small geospatial lookup for the planning applications in
the sample dataset. It maps a site's postcode to planning features that may
need to be considered when making a recommendation, such as:

- conservation areas
- flood-risk zones
- green belt
- listed-building status
- heritage assets at risk

## Design choice

For simplicity and to keep this project self-contained, the postcode-to-
feature mappings are hardcoded in `POSTCODE_LOCATION_LOOKUP` in
[`geospatial.py`](geospatial.py). The lookup covers the fictional postcodes used
by the supplied cases; it is not a general postcode or geospatial service.

This approach makes the example deterministic, fast, and usable without network
access or API credentials. It also means that the returned features should be
treated as fixture data, not as authoritative planning data.
An unknown postcode returns `found: false` rather than guessing or silently
returning an empty set of constraints.

In a production system, this mapping would be replaced with maintained spatial
datasets and point-in-polygon queries using the application's coordinates. Data
provenance, publication dates, refresh schedules, and the geometry used for each
match would also be retained so that an officer can verify the result.

## Usage

Call `postcode_lookup` with a postcode:

```python
from tools.geospatial import postcode_lookup

result = postcode_lookup("ZZ46 0BG")
```

Postcodes are matched case-insensitively and whitespace is ignored. A known
postcode returns the original trimmed input and its constraints:

```python
{
    "found": True,
    "postcode": "ZZ46 0BG",
    "constraints": {
        "conservation_area_name": None,
        "flood_risk_level": [2],
        "is_green_belt": False,
        "listed_building_grade": None,
        "heritage_at_risk_name": None,
    },
}
```

An unknown postcode returns:

```python
{
    "found": False,
    "postcode": "DN1 1AA",
    "constraints": None,
}
```

`found: true` with empty or false-valued constraints means that the postcode is
included in the fixtures but has none of the modelled features. It is
different from `found: false`, which means the postcode is outside the lookup's
limited coverage.
