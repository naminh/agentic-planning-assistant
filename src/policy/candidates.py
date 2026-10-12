"""Offline candidate generation: rank every policy/paragraph for each issue.

Ranks Local Plan policies and NPPF paragraphs against each issue's query with the
existing BM25 machinery, so a person can approve additions to `issues.yaml` instead of
relying on memory. Also flags current recipe entries that rank poorly. No LLM calls.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from domain import LOCAL_PLAN, NPPF, Issue, PolicyChunk
from policy.config import IssueRecipe
from policy.index import PolicyIndex

MAX_NPPF_PARAGRAPH = 230
SNIPPET_CHARS = 700
POLICY_NUMBER = re.compile(r"^Policy (\d+):")


@dataclass(frozen=True)
class Candidate:
    """One ranked policy or paragraph for an issue."""

    kind: str  # "local_plan" or "nppf"
    number: int
    title: str
    rank: int  # 1 = best match
    in_recipe: bool


def build_catalogue(index: PolicyIndex) -> list[PolicyChunk]:
    """One entry per Local Plan policy and per NPPF paragraph (heading + opening)."""
    catalogue: list[PolicyChunk] = []
    numbers = sorted(
        {
            int(m.group(1))
            for c in index.chunks
            if c.source == LOCAL_PLAN
            and c.policy_id
            and (m := POLICY_NUMBER.match(c.policy_id))
        }
    )
    for number in numbers:
        heading = index.local_plan_heading(number) or f"Policy {number}"
        chunks = index.policy_text(LOCAL_PLAN, f"Policy {number}:", limit=SNIPPET_CHARS)
        body = " ".join(c.text for c in chunks)
        catalogue.append(
            PolicyChunk(
                source=LOCAL_PLAN,
                page=chunks[0].page if chunks else 0,
                policy_id=heading,
                text=f"{heading}\n{body}"[: SNIPPET_CHARS + 120],
            )
        )
    for number in range(1, MAX_NPPF_PARAGRAPH + 1):
        found = index.nppf_paragraph_text(number, SNIPPET_CHARS)
        if found:
            catalogue.append(
                PolicyChunk(
                    source=NPPF,
                    page=found[0],
                    policy_id=f"paragraph {number}",
                    text=found[1],
                )
            )
    return catalogue


def _number(chunk: PolicyChunk) -> int:
    match = re.search(r"\d+", chunk.policy_id or "")
    return int(match.group(0)) if match else 0


def rank_candidates(
    catalogue: PolicyIndex, recipe: IssueRecipe
) -> dict[str, list[Candidate]]:
    """Rank all catalogue entries for a recipe, separately for each kind."""
    query = f"{recipe.label} {recipe.query}"
    in_recipe = {
        "local_plan": set(recipe.local_plan_numbers),
        "nppf": set(recipe.nppf_numbers),
    }
    ranked: dict[str, list[Candidate]] = {}
    for kind, source in (("local_plan", LOCAL_PLAN), ("nppf", NPPF)):
        hits = catalogue.search(query, k=len(catalogue.chunks), sources={source})
        ranked[kind] = [
            Candidate(
                kind=kind,
                number=_number(chunk),
                title=(
                    (chunk.policy_id or "") if kind == "local_plan" else chunk.text[:90]
                ),
                rank=position,
                in_recipe=_number(chunk) in in_recipe[kind],
            )
            for position, chunk in enumerate(hits, start=1)
        ]
    return ranked


def _table(candidates: list[Candidate], top: int) -> list[str]:
    lines = ["| rank | ref | what it is | in recipe |", "|---|---|---|---|"]
    for c in candidates[:top]:
        label = f"Policy {c.number}" if c.kind == "local_plan" else f"para {c.number}"
        title = c.title.replace("|", "/").replace("\n", " ")
        lines.append(f"| {c.rank} | {label} | {title} | {'✓' if c.in_recipe else ''} |")
    return lines


def candidate_report(
    index: PolicyIndex, recipes: dict[Issue, IssueRecipe], top: int = 8
) -> str:
    """Markdown report: top candidates per issue plus recipe entries ranking below it."""
    catalogue = PolicyIndex(build_catalogue(index))
    lines = [
        "# Policy candidates (offline, for human approval)",
        "",
        "Each policy / NPPF paragraph is ranked by BM25 against the issue's label and "
        f"query. ✓ marks entries already in `issues.yaml`. Top {top} shown; approve any "
        "new ones by adding them (with an `expect` keyword) to the recipe.",
    ]
    for issue, recipe in recipes.items():
        ranked = rank_candidates(catalogue, recipe)
        lines += ["", f"## {issue.value}", "", f"Query: _{recipe.query}_", ""]
        for kind, heading in (("local_plan", "Local Plan"), ("nppf", "NPPF")):
            lines += [f"### {heading}", "", *_table(ranked[kind], top), ""]
            label = "Policy" if kind == "local_plan" else "para"
            wanted = (
                recipe.local_plan_numbers
                if kind == "local_plan"
                else recipe.nppf_numbers
            )
            matched = {c.number for c in ranked[kind]}
            review = [
                f"- {label} {c.number} is rank {c.rank} of {len(ranked[kind])}"
                for c in ranked[kind]
                if c.in_recipe and c.rank > top
            ] + [
                f"- {label} {n} shares no keywords with the query"
                for n in sorted(set(wanted) - matched)
            ]
            if review:
                lines += [
                    f"Recipe entries to review (below the top {top}):",
                    *review,
                    "",
                ]
    return "\n".join(lines) + "\n"


def apply_candidates(
    config: dict[str, Any], catalogue: PolicyIndex, top: int = 5
) -> list[tuple[str, str, int]]:
    """Append each issue's top-`top` ranked candidates that its recipe lacks.

    Works on the parsed `issues.yaml` dict, adding entries flagged `generated: true`
    after the hand-written ones. Idempotent: ranking depends only on the issue's label
    and query, so a second call finds those candidates already present.
    Returns the (issue, kind, number) triples that were added.
    """
    added: list[tuple[str, str, int]] = []
    for issue_id, entry in config["issues"].items():
        recipe = IssueRecipe.model_validate(entry)
        ranked = rank_candidates(catalogue, recipe)
        for kind, key, field_name in (
            ("local_plan", "policy", "local_plan"),
            ("nppf", "para", "nppf"),
        ):
            for candidate in ranked[kind][:top]:
                if not candidate.in_recipe:
                    entry.setdefault(field_name, []).append(
                        {key: candidate.number, "generated": True}
                    )
                    added.append((issue_id, kind, candidate.number))
    return added


def apply_candidates_file(
    path: Path, index: PolicyIndex, top: int = 5
) -> list[tuple[str, str, int]]:
    """`apply_candidates` on an `issues.yaml` file, rewriting it in place."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    added = apply_candidates(config, PolicyIndex(build_catalogue(index)), top)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=110),
        encoding="utf-8",
    )
    return added
