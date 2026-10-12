"""Case-specific search queries, written once per case by the cheap model.

Each issue has a fixed recipe query. It cannot know that this proposal is a barn
conversion beside a watercourse, so the model proposes one or two extra queries per issue
in the vocabulary the policy text would use. They are searched alongside the recipe's
query; if the call fails the recipe queries are used alone, so this step can only add.
"""

from loguru import logger
from pydantic import BaseModel, Field

from domain import Issue
from llm import LLMClient
from pipeline.models import CaseFacts
from policy.config import PolicyConfig

MAX_QUERIES_PER_ISSUE = 2
MAX_QUERY_CHARS = 160

QUERY_SYSTEM = (
    "You help a planning officer search the policy documents for one application. "
    "For each listed issue write one or two short search queries (6 to 12 words) "
    "using the wording planning policy itself would use (for example 'sequential "
    "test' rather than 'is it safe from flooding'), tailored to this proposal's type, "
    "site and constraints. Use only the facts given; add no new facts. Do not repeat "
    "the issue's baseline query."
)


class IssueQueries(BaseModel):
    """The extra queries proposed for one issue."""

    issue: Issue
    queries: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """What the query-writing model returns."""

    plans: list[IssueQueries] = Field(default_factory=list)


def clean_queries(plan: QueryPlan, issues: list[Issue]) -> dict[Issue, list[str]]:
    """Keep sane queries for the requested issues only (the model is not trusted)."""
    wanted = set(issues)
    out: dict[Issue, list[str]] = {}
    for item in plan.plans:
        if item.issue not in wanted:
            continue
        queries = [
            " ".join(q.split())[:MAX_QUERY_CHARS]
            for q in item.queries
            if isinstance(q, str) and len(q.split()) >= 2
        ]
        if queries:
            out[item.issue] = list(dict.fromkeys(queries))[:MAX_QUERIES_PER_ISSUE]
    return out


async def expand_queries(
    client: LLMClient,
    model: str,
    facts: CaseFacts,
    issues: list[Issue],
    config: PolicyConfig,
    site_note: str = "",
) -> dict[Issue, list[str]]:
    """Extra search queries per issue; empty if the model cannot be reached.

    Args:
        client: LLM client (cached, so a rerun costs nothing).
        model: Model used to write the queries (the cheap tier).
        facts: Structured facts of the case; the raw pack is not shown to this call.
        issues: Issues being assessed.
        config: Issue recipes, used to show the model each issue's baseline query.
        site_note: What is known about the site's constraints.
    """
    listing = "\n".join(
        f"- {issue.value}: baseline query: {config.recipes[issue].query}"
        for issue in issues
    )
    user = (
        f"PROPOSAL: {facts.proposal}\nTYPE: {facts.development_type}\n"
        f"EXISTING USE: {facts.existing_use}\nPROPOSED USE: {facts.proposed_use}\n"
        f"FLAGS: {', '.join(facts.flags) or 'none'}\nSITE: {site_note}\n\n"
        f"ISSUES:\n{listing}"
    )
    try:
        plan = await client.parse(
            QueryPlan, model, QUERY_SYSTEM, user, max_tokens=2000, stage="queries"
        )
    except Exception as err:  # best effort: the recipe queries still run
        logger.warning(f"Query expansion skipped: {type(err).__name__}: {err}")
        return {}
    return clean_queries(plan, issues)
