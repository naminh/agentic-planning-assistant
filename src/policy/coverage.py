"""Retrieval that checks its own coverage and reformulates at most twice.

After the first retrieval, the distinctive words of the issue's queries are compared with
the extracts found (stemmed, so 'drained' covers 'drainage'). If too many are absent,
the cheap model is shown the missing words and writes one or two alternative queries in
policy wording, and retrieval runs again with them added. The loop is bounded and its
stopping rule is deterministic: it ends when coverage reaches the target, after
`MAX_ATTEMPTS` reformulations, when the model proposes nothing new, or when a retry
finds no new chunk. It can only add extracts, never remove one, and a failed model call
ends it with the first result. Grounding, the guard and the verifier stay single-shot.

With Jev on (`coverage` in JEV_STAGES), a gap in the word count only triggers the model
if Jev, asked per query whether the extracts address it, is not confident that they all
do. That spares the reformulation call when the policy is in the extracts under other
words. Jev can only suppress a reformulation, never add or remove an extract.
"""

from collections.abc import Sequence

from loguru import logger
from pydantic import BaseModel, Field

from domain import Issue, PolicyChunk
from jev import JevClient, JevError, noul
from llm import CacheMissError, LLMClient
from policy.config import PolicyConfig
from policy.fuzzy import stem
from policy.index import PolicyIndex, tokenize
from policy.retrieval import policy_block, retrieve_policy
from tools.geospatial import LocationConstraints

MAX_ATTEMPTS = 2
COVERAGE_TARGET = 0.7
MIN_TERM_LEN = 4  # shorter words are too generic to count as coverage
MAX_QUERY_CHARS = 160
MAX_NEW_QUERIES = 2
JEV_EXTRACT_CHARS = 60_000  # Jev reads up to 32k tokens: leave room for the questions

REFORMULATE_SYSTEM = (
    "You help a planning officer search policy documents. Earlier searches for an "
    "issue did not turn up text containing some of the listed words. Write one or two "
    "new short search queries (6 to 12 words) that reach those ideas using the "
    "wording policy itself would use, including synonyms. Add no facts about the "
    "case."
)


class Reformulation(BaseModel):
    """New queries proposed after a coverage gap."""

    queries: list[str] = Field(default_factory=list)


def query_terms(queries: Sequence[str]) -> set[str]:
    """Distinctive stems in the queries."""
    return {stem(t) for q in queries for t in tokenize(q) if len(t) >= MIN_TERM_LEN}


def coverage(
    queries: Sequence[str], chunks: Sequence[PolicyChunk]
) -> tuple[float, list[str]]:
    """Fraction of the queries' distinctive stems found in the extracts, and the rest."""
    terms = query_terms(queries)
    if not terms:
        return 1.0, []
    seen = {stem(t) for c in chunks for t in tokenize(c.text)}
    missing = sorted(terms - seen)
    return 1 - len(missing) / len(terms), missing


def clean(queries: Sequence[str], tried: Sequence[str]) -> list[str]:
    """Sane, new queries only (the model is not trusted)."""
    known = {" ".join(q.lower().split()) for q in tried}
    out: list[str] = []
    for q in queries:
        text = " ".join(str(q).split())[:MAX_QUERY_CHARS]
        if len(text.split()) >= 2 and text.lower() not in known:
            known.add(text.lower())
            out.append(text)
    return out[:MAX_NEW_QUERIES]


async def jev_addresses(
    jev: JevClient, chunks: Sequence[PolicyChunk], queries: Sequence[str]
) -> bool:
    """True if Jev is confident the extracts address every query (False otherwise)."""
    questions = {
        str(n): noul(
            {
                "query": q,
                "question": "Do the policy extracts in the state contain wording that "
                "addresses what `query` is looking for?",
            },
            yes="Some extract states a policy, requirement or guidance on the subject "
            "of the query.",
            no="No extract deals with the subject of the query.",
        )
        for n, q in enumerate(queries)
    }
    try:
        result = await jev.decide(
            policy_block(list(chunks))[:JEV_EXTRACT_CHARS], questions, stage="coverage"
        )
    except JevError as err:
        logger.warning(f"Jev coverage check failed ({err}); using the word count")
        return False
    tau = jev.tau("coverage")
    return all(result.noul(k).lean(tau) is True for k in questions)


async def retrieve_with_coverage(
    client: LLMClient,
    model: str | None,
    index: PolicyIndex,
    config: PolicyConfig,
    issue: Issue,
    development_type: str,
    constraints: LocationConstraints | None = None,
    *,
    extra_queries: Sequence[str] = (),
    jev: JevClient | None = None,
) -> list[PolicyChunk]:
    """`retrieve_policy`, widened (not at all when `model` is None), by up to `MAX_ATTEMPTS` coverage-driven reformulations."""
    recipe = config.recipes[issue]
    base = [f"{recipe.query} {development_type}"]
    extra = list(extra_queries)

    def run() -> list[PolicyChunk]:
        return retrieve_policy(
            index, config, issue, development_type, constraints, extra_queries=extra
        )

    chunks = run()
    if model is None:
        return chunks
    for attempt in range(1, MAX_ATTEMPTS + 1):
        score, missing = coverage([*base, *extra], chunks)
        if score >= COVERAGE_TARGET:
            break
        if (
            jev is not None
            and jev.active("coverage")
            and await jev_addresses(jev, chunks, [*base, *extra])
        ):
            logger.info(f"{issue.value}: coverage {score:.0%} but Jev finds it covered")
            break
        try:
            proposal = await client.parse(
                Reformulation,
                model,
                REFORMULATE_SYSTEM,
                f"ISSUE: {recipe.label}\nPROPOSAL TYPE: {development_type}\n"
                f"EARLIER QUERIES: {'; '.join([recipe.query, *extra])}\n"
                f"WORDS NOT FOUND IN ANY EXTRACT: {', '.join(missing[:20])}",
                max_tokens=600,
                stage=f"reformulate:{issue.value}:{attempt}",
            )
        except CacheMissError:
            raise
        except Exception as err:  # a flaky helper call must not fail the issue
            logger.warning(f"Reformulation failed for {issue.value} ({err}); stopping")
            break
        new = clean(proposal.queries, [recipe.query, *base, *extra])
        if not new:
            break
        extra += new
        widened = run()
        if len(widened) <= len(chunks):
            break
        logger.info(
            f"{issue.value}: coverage {score:.0%}, attempt {attempt} added "
            f"{len(widened) - len(chunks)} extracts"
        )
        chunks = widened
    return chunks
