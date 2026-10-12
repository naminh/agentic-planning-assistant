"""Config loading, reference resolution and routing data (offline; no API calls)."""

from pathlib import Path

import pytest
import yaml

from config import load_settings
from domain import LOCAL_PLAN, Issue, PolicyChunk
from policy.config import load_config
from policy.index import PolicyIndex

REPO = Path(__file__).resolve().parent.parent
HAS_DATA = (
    REPO / "data/extracted/local_policy/Doncaster_Local_Plan_Adopted_23_Sept_2021.json"
).exists()


CONFIG = load_config()
CONFIG_DIR = REPO / "configs/doncaster"


def write_config(tmp_path: Path, **overrides: dict[str, object]) -> Path:
    """A copy of the real config folder with `sources`/`issues` YAML replaced."""
    for name in ("sources", "issues"):
        data = overrides.get(name) or yaml.safe_load(
            (CONFIG_DIR / f"{name}.yaml").read_text("utf-8")
        )
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------ config loading


def test_every_issue_has_a_recipe() -> None:
    assert set(CONFIG.recipes) == set(Issue)


def test_core_issues_and_keywords_loaded_in_order() -> None:
    routing = CONFIG.routing
    assert routing.core[0] == Issue.PRINCIPLE
    assert routing.keyword_rules[0].issue == Issue.FLOOD  # file order is preserved
    proposal_only = {r.issue for r in routing.keyword_rules if r.proposal_only}
    assert Issue.HMO in proposal_only and Issue.FLOOD not in proposal_only


def test_sources_carry_versions_and_groups() -> None:
    assert all(s.version for s in CONFIG.sources)
    assert len(CONFIG.legislation) == 4
    assert "bessacarr" in CONFIG.conservation_area_sources
    assert CONFIG.nppf_edition == "December 2024"


def test_versions_for_matches_longest_label_prefix() -> None:
    cites = ["NPPF p.39 (paragraphs 134-135)", "Doncaster Local Plan p.143 (Policy 44)"]
    assert CONFIG.versions_for(cites) == {
        "Doncaster Local Plan": "Adopted 23 September 2021",
        "NPPF": "December 2024",
    }


def _issues_yaml() -> dict[str, dict[str, object]]:
    return yaml.safe_load((CONFIG_DIR / "issues.yaml").read_text("utf-8"))  # type: ignore[no-any-return]


def _sources_yaml() -> dict[str, object]:
    return yaml.safe_load((CONFIG_DIR / "sources.yaml").read_text("utf-8"))  # type: ignore[no-any-return]


def test_unknown_issue_id_fails_fast(tmp_path: Path) -> None:
    good = _issues_yaml()
    good["issues"]["not_an_issue"] = {"label": "x", "query": "x"}  # type: ignore[index]
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, issues=good))


def test_unknown_extra_source_fails_fast(tmp_path: Path) -> None:
    good = _issues_yaml()
    good["issues"]["principle_of_development"]["extra_sources"] = ["No Such Source"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown extra_sources"):
        load_config(write_config(tmp_path, issues=good))


def test_duplicate_source_labels_fail_fast(tmp_path: Path) -> None:
    entry = {"label": "A", "path": "a.json"}
    sources = {"council": "X", "sources": [entry, entry]}
    with pytest.raises(ValueError, match="duplicate source labels"):
        load_config(write_config(tmp_path, sources=sources))


def test_unknown_field_in_sources_fails_fast(tmp_path: Path) -> None:
    entry = {"label": "A", "path": "a.json", "colour": "red"}
    sources = {"council": "X", "sources": [entry]}
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, sources=sources))


# ------------------------------------------------------------- reference resolvers


def synthetic_nppf() -> PolicyIndex:
    contents = "Contents\n11. Making effective use of land 36\n12. Achieving well-designed places 38"
    chapter = "11. Making effective use of land\n124. Planning policies and decisions should promote an effective use of land in meeting the need for homes."
    real = (
        "11. Plans and decisions should apply a presumption in favour of sustainable "
        "development. For plan-making this means that:\n12. The presumption does not "
        "change the statutory status of the development plan."
    )
    return PolicyIndex(
        [
            PolicyChunk(source="NPPF", page=3, text=contents),
            PolicyChunk(source="NPPF", page=6, text=real),
            PolicyChunk(source="NPPF", page=36, text=chapter),
        ]
    )


def test_nppf_resolver_skips_contents_and_chapter_headings() -> None:
    page, text = synthetic_nppf().nppf_paragraph_text(11) or (0, "")
    assert page == 6 and text.startswith("11. Plans and decisions should apply")
    assert "12." not in text  # stops at the next paragraph


def test_nppf_resolver_accepts_short_colon_openings() -> None:
    index = PolicyIndex(
        [
            PolicyChunk(
                source="NPPF",
                page=33,
                text="117. Within this context, applications for development should:\na) give priority",
            )
        ]
    )
    assert index.nppf_paragraph_text(117) is not None


def test_nppf_resolver_returns_none_for_missing_paragraph() -> None:
    assert synthetic_nppf().nppf_paragraph_text(999) is None


def test_local_plan_heading_lookup() -> None:
    index = PolicyIndex(
        [
            PolicyChunk(
                source=LOCAL_PLAN,
                page=163,
                policy_id="Policy 57: Flood Risk Management",
                text="Flood zone sequential test exception test drainage",
            )
        ]
    )
    assert index.local_plan_heading(57) == "Policy 57: Flood Risk Management"
    assert index.local_plan_heading(5) is None  # 'Policy 5' must not match 'Policy 57'


# --------------------------------------- every curated reference, on the real corpus


@pytest.fixture(scope="module")
def real_index() -> PolicyIndex:
    return PolicyIndex.load(load_settings(require_key=False), CONFIG)


def _references() -> list[tuple[str, str, int, str | None, bool]]:
    rows = []
    for issue, recipe in CONFIG.recipes.items():
        for kind, refs in (("local_plan", recipe.local_plan), ("nppf", recipe.nppf)):
            rows += [(issue.value, kind, r.number, r.expect, r.generated) for r in refs]
    return rows


@pytest.mark.skipif(not HAS_DATA, reason="run `uv run parser` first")
@pytest.mark.parametrize(
    ("issue", "kind", "number", "expect", "generated"),
    _references(),
    ids=lambda v: str(v),
)
def test_curated_reference_resolves_and_matches_its_topic(
    real_index: PolicyIndex,
    issue: str,
    kind: str,
    number: int,
    expect: str | None,
    generated: bool,
) -> None:
    # Hand-written references must declare what they are about; generated ones (added
    # by keyword ranking) only have to exist in the corpus.
    if not generated:
        assert expect, f"{issue}: {kind} {number} must declare an `expect` keyword"
    if kind == "local_plan":
        found = real_index.local_plan_heading(number)
        assert found, f"{issue}: Local Plan Policy {number} not found"
        if expect:
            assert (
                expect.lower() in found.lower()
            ), f"{issue}: Policy {number} is {found!r}"
    else:
        resolved = real_index.nppf_paragraph_text(number, limit=3000)
        assert resolved, f"{issue}: NPPF paragraph {number} not found"
        assert (
            not expect or expect.lower() in resolved[1].lower()
        ), f"{issue}: NPPF paragraph {number} reads {resolved[1][:120]!r}"
