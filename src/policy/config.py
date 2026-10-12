"""Policy configuration for one council: sources, issue recipes and routing rules.

Everything is read from `configs/<council>/{sources,issues}.yaml` (set
`PLANNING_CONFIG_DIR` to use another council's folder) and validated on load, so a bad
file fails fast. `load_config()` returns one immutable `PolicyConfig` that is passed
explicitly to the index, retrieval and routing code; nothing is read at import time.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from domain import DocKind, Issue

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "doncaster"
LEGISLATION_GROUP = "legislation"


def config_dir() -> Path:
    """Folder holding `sources.yaml` and `issues.yaml` for the active council."""
    override = os.getenv("PLANNING_CONFIG_DIR")
    return Path(override) if override else DEFAULT_CONFIG_DIR


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicySource(_Strict):
    """One parsed PDF and how it should be indexed.

    `default_search=False` keeps very large documents out of general searches; they
    are only searched when a recipe names them (`extra_sources`, `statute_query`) or
    the site's conservation area selects them. `ocr=True` marks scanned documents
    whose text must be transcribed by a vision model (see `policy/ocr.py`).
    `version` records the edition so citations can be audited later.
    """

    label: str
    path: str  # relative to data/extracted, as written by `uv run parser`
    version: str = "unspecified"
    published: str | None = None
    url: str | None = None
    default_search: bool = True
    ocr: bool = False
    captions: bool = False  # describe large figures/maps with a vision model (opt-in)
    group: str | None = None
    conservation_area: str | None = None

    @field_validator("version", "published", mode="before")
    @classmethod
    def _stringify(cls, value: object) -> object:
        return None if value is None else str(value)

    @property
    def stem(self) -> str:
        """File name without extension (used for OCR cache files)."""
        return self.path.rsplit("/", 1)[-1].removesuffix(".json")


class PolicyRef(_Strict):
    """A Local Plan policy (`policy:`) or NPPF paragraph (`para:`) a recipe names.

    `expect` is a keyword the reference's heading/text must contain (checked by tests).
    `generated` entries were added by `scripts/suggest_policies.py --apply`, ranked by
    keyword match rather than hand-checked: tests only require that they resolve.
    """

    number: int = Field(validation_alias=AliasChoices("policy", "para"))
    expect: str | None = None
    generated: bool = False


class EvidenceRequirement(_Strict):
    """A document an officer needs in the pack before this issue can be decided.

    It is met when any document is classified (by content, see `pipeline.doctype`)
    as one of `kinds`; a passing mention in another document does not count, so a
    missing report is surfaced for the officer rather than assumed. `applies_if`
    limits it to sites with a constraint (`flood_zone` or `heritage_designation`);
    an unknown site never triggers a conditional requirement.
    """

    label: str
    kinds: list[DocKind]
    applies_if: Literal["flood_zone", "heritage_designation"] | None = None


class IssueRecipe(_Strict):
    """What to retrieve when an issue is in play."""

    label: str
    query: str
    required_evidence: list[EvidenceRequirement] = []
    local_plan: list[PolicyRef] = []
    nppf: list[PolicyRef] = []
    extra_sources: list[str] = []
    statute_query: str = ""  # searched only within the `legislation` group when set
    search_k: int = 4

    @property
    def local_plan_numbers(self) -> tuple[int, ...]:
        """Policy numbers, in recipe order."""
        return tuple(r.number for r in self.local_plan)

    @property
    def nppf_numbers(self) -> tuple[int, ...]:
        """Paragraph numbers, in recipe order."""
        return tuple(r.number for r in self.nppf)


class KeywordRule(_Strict):
    """Signals that an issue is in play.

    They are shown to the routing model as hints and, if that call fails, matched
    against the case text. A keyword matches whole words (plus a simple plural);
    a trailing `*` makes it a prefix (`ecolog*` matches "ecology", "ecological").
    """

    issue: Issue
    keywords: list[str]
    proposal_only: bool = False  # match the proposal text only, not consultee replies

    @field_validator("keywords")
    @classmethod
    def _non_blank(cls, keywords: list[str]) -> list[str]:
        if any(not k.strip().removesuffix("*").strip() for k in keywords):
            raise ValueError("keywords must not be blank")
        return keywords


class Routing(_Strict):
    """Which issues a case triggers: the core set plus keyword-matched ones."""

    core: list[Issue]
    keyword_rules: list[KeywordRule]

    def hints_for(self, issue: Issue) -> tuple[str, ...]:
        """Plain-text signals for `issue` (stems without the `*`); empty if none."""
        return tuple(
            k.removesuffix("*")
            for r in self.keyword_rules
            if r.issue == issue
            for k in r.keywords
        )


class _IssuesFile(_Strict):
    council: str
    nppf_edition: str
    routing: Routing
    issues: dict[Issue, IssueRecipe]


class _SourcesFile(_Strict):
    council: str
    sources: list[PolicySource]


@dataclass(frozen=True)
class PolicyConfig:
    """Validated sources, recipes and routing for one council."""

    council: str
    nppf_edition: str
    sources: tuple[PolicySource, ...]
    recipes: dict[Issue, IssueRecipe]
    routing: Routing

    def labels_in_group(self, group: str) -> tuple[str, ...]:
        """Labels of the sources tagged with `group`, in file order."""
        return tuple(s.label for s in self.sources if s.group == group)

    @property
    def legislation(self) -> tuple[str, ...]:
        """Labels of the statute sources (searched only via a recipe's `statute_query`)."""
        return self.labels_in_group(LEGISLATION_GROUP)

    @property
    def opt_in_labels(self) -> frozenset[str]:
        """Sources excluded from general searches (`default_search: false`)."""
        return frozenset(s.label for s in self.sources if not s.default_search)

    @property
    def conservation_area_sources(self) -> dict[str, tuple[str, ...]]:
        """`LocationConstraints.conservation_area_name` -> appraisal source labels."""
        areas: dict[str, tuple[str, ...]] = {}
        for source in self.sources:
            if source.conservation_area:
                areas[source.conservation_area] = (
                    *areas.get(source.conservation_area, ()),
                    source.label,
                )
        return areas

    def versions_for(self, citations: Iterable[str]) -> dict[str, str]:
        """Map each source named in `citations` ("<label> p.N ...") to its version."""
        by_length = sorted(self.sources, key=lambda s: len(s.label), reverse=True)
        seen: dict[str, str] = {}
        for citation in citations:
            for source in by_length:
                if citation.startswith(source.label):
                    seen.setdefault(source.label, source.version)
                    break
        return dict(sorted(seen.items()))


def _read(path: Path) -> object:
    return yaml.safe_load(path.read_text("utf-8"))


def _build(sources: _SourcesFile, issues: _IssuesFile) -> PolicyConfig:
    labels = [s.label for s in sources.sources]
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        raise ValueError(f"duplicate source labels: {duplicates}")
    for issue, recipe in issues.issues.items():
        unknown = [s for s in recipe.extra_sources if s not in labels]
        if unknown:
            raise ValueError(f"{issue.value}: unknown extra_sources {unknown}")
    return PolicyConfig(
        council=issues.council,
        nppf_edition=issues.nppf_edition,
        sources=tuple(sources.sources),
        recipes=issues.issues,
        routing=issues.routing,
    )


@lru_cache
def _load(directory: Path) -> PolicyConfig:
    return _build(
        _SourcesFile.model_validate(_read(directory / "sources.yaml")),
        _IssuesFile.model_validate(_read(directory / "issues.yaml")),
    )


def load_config(directory: Path | None = None) -> PolicyConfig:
    """Load (and cache) the config in `directory`, default `config_dir()`."""
    return _load(directory or config_dir())
