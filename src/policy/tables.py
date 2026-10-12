"""Rebuild flattened policy tables as markdown, accepting only a faithful rebuild.

The PDF extractor emits a table's cells one per line, so rows and columns are lost and a
threshold can sit far from its heading. A model can usually put the rows back, but a
model can also drop a row or change a number, which in a policy table is worse than
leaving it flat. So its output is accepted only if, word for word, it adds nothing the
source did not contain and drops almost nothing it did. Anything else keeps the flat text.
"""

import asyncio
import re

from loguru import logger

from domain import PolicyChunk
from llm import LLMClient

MAX_TABLE_CHARS = 4500  # larger tables are left flat (and their parts are not rebuilt)
MIN_COVERAGE = 0.95  # share of the source's tokens that must survive the rebuild
GIVE_UP = "CANNOT_STRUCTURE"

TABLE_SYSTEM = (
    "The text below is a table from a planning policy document, flattened so that cells "
    "appear one per line in reading order and the row structure is lost. Rebuild it as a "
    "GitHub markdown table, keeping the caption as a line above it. Use ONLY the words "
    "and numbers in the text: do not add, infer, correct, translate or omit anything, "
    "and keep every number and unit exactly as written. If the rows cannot be recovered "
    f"with confidence, reply with exactly {GIVE_UP}. Reply with the table only."
)

_TOKEN = re.compile(r"[a-z0-9]+")
_FENCE = re.compile(r"^```(?:markdown|md)?\s*|\s*```$", re.MULTILINE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def faithful(source: str, markdown: str) -> bool:
    """True if `markdown` is the same content as `source`, only rearranged.

    Every token in the rebuild must occur in the source (nothing invented), and at
    least `MIN_COVERAGE` of the source's tokens must occur in the rebuild (nothing
    dropped). Markdown punctuation and the separator row carry no tokens.
    """
    source_tokens = _tokens(source)
    rebuilt = _tokens(markdown)
    if not source_tokens or not rebuilt:
        return False
    source_set = set(source_tokens)
    if any(t not in source_set for t in rebuilt):
        return False
    rebuilt_set = set(rebuilt)
    kept = sum(1 for t in source_tokens if t in rebuilt_set)
    return kept / len(source_tokens) >= MIN_COVERAGE


async def _refine_one(
    client: LLMClient, model: str, chunk: PolicyChunk, reasoning_off: bool = False
) -> PolicyChunk:
    try:
        reply = await client.complete(
            model,
            TABLE_SYSTEM,
            chunk.text,
            max_tokens=4000,
            stage="table",
            reasoning_off=reasoning_off,
        )
    except Exception as err:  # the flat text is still usable
        logger.warning(f"Table left flat ({chunk.heading}): {type(err).__name__}")
        return chunk
    markdown = _FENCE.sub("", reply).strip()
    if markdown == GIVE_UP or not faithful(chunk.text, markdown):
        logger.info(f"Table left flat (rebuild not faithful): {chunk.heading}")
        return chunk
    return chunk.model_copy(update={"text": markdown})


async def refine_tables(
    client: LLMClient,
    model: str,
    chunks: list[PolicyChunk],
    reasoning_off: bool = False,
) -> list[PolicyChunk]:
    """Replace each whole, reasonably sized table chunk by a faithful markdown rebuild."""
    targets = [
        i
        for i, c in enumerate(chunks)
        if c.kind == "table" and c.parts == 1 and len(c.text) <= MAX_TABLE_CHARS
    ]
    rebuilt = await asyncio.gather(
        *(_refine_one(client, model, chunks[i], reasoning_off) for i in targets)
    )
    out = list(chunks)
    for i, chunk in zip(targets, rebuilt, strict=True):
        out[i] = chunk
    changed = sum(
        1 for i, c in zip(targets, rebuilt, strict=True) if c is not chunks[i]
    )
    logger.info(f"Tables rebuilt as markdown: {changed} of {len(targets)}")
    return out
