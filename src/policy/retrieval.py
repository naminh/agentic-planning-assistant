"""Choose the policy extracts the assessor sees for one issue."""

from collections.abc import Sequence

from domain import LOCAL_PLAN, Issue, PolicyChunk
from policy.config import IssueRecipe, PolicyConfig
from policy.index import PolicyIndex
from tools.geospatial import LocationConstraints

POLICY_CHARS = 3000  # per named Local Plan policy, for its supporting text and tables
WHOLE_UNIT_KINDS = ("policy", "table")  # split units whose missing parts would mislead
STATUTE_K = 3
EXTRA_SOURCES_K = 2
CONSERVATION_K = {
    Issue.HERITAGE: 3,
    Issue.DESIGN: 2,
}  # issues that read area appraisals


def retrieve_policy(
    index: PolicyIndex,
    config: PolicyConfig,
    issue: Issue,
    development_type: str,
    constraints: LocationConstraints | None = None,
    *,
    use_curated: bool = True,
    use_search: bool = True,
    hybrid: bool = True,
    extra_queries: Sequence[str] = (),
) -> list[PolicyChunk]:
    """The issue's named policies plus case-specific keyword hits, de-duplicated.

    Args:
        index: Policy index to look up and search.
        config: Issue recipes (named policies, search query, extra sources).
        issue: The issue being assessed.
        development_type: Extracted proposal type, appended to the search query.
        constraints: Site constraints; they gate site-specific sources such as
            conservation area appraisals. None means no site-specific sources.
        use_curated: Include the recipe's named Local Plan policies and NPPF
            paragraphs. Switch off for ablations.
        use_search: Include everything found by search and the extra sources.
            Switch off for ablations.
        hybrid: Add the fuzzy stem/trigram ranking's extra hits to BM25's. Off means
            BM25 alone, for ablations.
        extra_queries: Further queries for this issue (see `policy.queries`); each is
            searched alongside the recipe's own and the rankings are fused.

    Returns:
        De-duplicated chunks, each carrying its citation.
    """
    recipe = config.recipes[issue]
    chunks: list[PolicyChunk] = []
    if use_curated:
        chunks += _named_policies(index, recipe)
    if use_search:
        query = f"{recipe.query} {development_type}"
        queries = [query, *extra_queries] if hybrid else [query]
        found = _search(index, queries, recipe.search_k, None, hybrid)
        found += _extra_sources(index, recipe, queries, hybrid)
        found += _conservation_area(index, config, issue, queries, constraints, hybrid)
        found += _statute(index, config, recipe, development_type, hybrid)
        chunks += _with_whole_units(index, found)
    return _unique(chunks)


def policy_block(chunks: list[PolicyChunk]) -> str:
    """The extracts as they appear in the prompt, each headed by its citation."""
    return "\n\n".join(f"[{c.citation}]\n{c.text}" for c in chunks)


def _named_policies(index: PolicyIndex, recipe: IssueRecipe) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    for number in recipe.local_plan_numbers:
        chunks += index.policy_text(LOCAL_PLAN, f"Policy {number}:", limit=POLICY_CHARS)
    for paragraph in recipe.nppf_numbers:
        chunks += index.nppf_paragraph(paragraph)
    return chunks


def _search(
    index: PolicyIndex,
    queries: Sequence[str],
    k: int,
    sources: set[str] | None,
    hybrid: bool,
) -> list[PolicyChunk]:
    if hybrid:
        return index.hybrid_search(queries, k, sources)
    return index.search(queries[0], k, sources)


def _extra_sources(
    index: PolicyIndex, recipe: IssueRecipe, queries: Sequence[str], hybrid: bool
) -> list[PolicyChunk]:
    """Documents the recipe asks for by name (e.g. SPDs kept out of default search)."""
    if not recipe.extra_sources:
        return []
    return _search(index, queries, EXTRA_SOURCES_K, set(recipe.extra_sources), hybrid)


def _conservation_area(
    index: PolicyIndex,
    config: PolicyConfig,
    issue: Issue,
    queries: Sequence[str],
    constraints: LocationConstraints | None,
    hybrid: bool,
) -> list[PolicyChunk]:
    """Appraisals only for the conservation area the site is actually in."""
    area = constraints.conservation_area_name if constraints else None
    if not area or issue not in CONSERVATION_K:
        return []
    labels = config.conservation_area_sources.get(area, ())
    if not labels:
        return []
    return _search(index, queries, CONSERVATION_K[issue], set(labels), hybrid)


def _statute(
    index: PolicyIndex,
    config: PolicyConfig,
    recipe: IssueRecipe,
    development_type: str,
    hybrid: bool,
) -> list[PolicyChunk]:
    """Legislation is searched only through the recipe's explicit `statute_query`."""
    if not recipe.statute_query:
        return []
    query = f"{recipe.statute_query} {development_type}"
    return _search(index, [query], STATUTE_K, set(config.legislation), hybrid)


def _with_whole_units(index: PolicyIndex, hits: list[PolicyChunk]) -> list[PolicyChunk]:
    """Add the other parts of any split policy or table a search hit landed in.

    Half a table (or half a policy's clauses) is worse than none: the assessor would
    apply a threshold without its conditions, or miss a row.
    """
    out: list[PolicyChunk] = []
    for hit in hits:
        out.append(hit)
        if hit.parts > 1 and hit.kind in WHOLE_UNIT_KINDS:
            out += index.unit_parts(hit)
    return out


def _unique(chunks: list[PolicyChunk]) -> list[PolicyChunk]:
    seen: set[tuple[str, int, str]] = set()
    unique = []
    for chunk in chunks:
        key = (chunk.source, chunk.page, chunk.text[:60])
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique
