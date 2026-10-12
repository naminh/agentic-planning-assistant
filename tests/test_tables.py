"""Offline tests for table rebuilding: only faithful rebuilds are accepted."""

import asyncio

from domain import PolicyChunk
from llm import LLMClient
from policy.tables import GIVE_UP, faithful, refine_tables

FLAT = "Table 13: Play area requirements\nLAP\nLEAP\nNEAP\nWalking distance\n100m\n400m\n1000m"
GOOD = (
    "Table 13: Play area requirements\n\n| | LAP | LEAP | NEAP |\n|---|---|---|---|\n"
    "| Walking distance | 100m | 400m | 1000m |"
)


class Scripted(LLMClient):
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    async def complete(  # type: ignore[override]
        self,
        model,
        system,
        user,
        images=None,
        max_tokens=4000,
        stage="llm",
        reasoning_off=False,
    ) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def table(text: str = FLAT, **kw: object) -> PolicyChunk:
    return PolicyChunk(
        source="S", page=1, kind="table", heading="Table 13", text=text, **kw
    )  # type: ignore[arg-type]


def test_a_faithful_rebuild_is_accepted() -> None:
    assert faithful(FLAT, GOOD)


def test_an_invented_or_changed_number_is_rejected() -> None:
    assert not faithful(FLAT, GOOD.replace("400m", "450m"))
    assert not faithful(FLAT, GOOD + "\n| Buffer | 5m | 20m | 30m |")


def test_a_dropped_row_is_rejected() -> None:
    long_flat = FLAT + "\nBuffer zone\n5m\n20m\n30m\nAge range\nUnder 5\n6 to 11\n8+"
    assert not faithful(long_flat, GOOD)


def test_refine_replaces_faithful_tables_and_keeps_the_rest() -> None:
    chunks = [table(), PolicyChunk(source="S", page=2, text="prose stays as it is")]
    out = asyncio.run(refine_tables(Scripted(GOOD), "m", chunks))
    assert out[0].text == GOOD and out[1].text == "prose stays as it is"
    assert out[0].kind == "table" and out[0].heading == "Table 13"


def test_refine_keeps_flat_text_when_the_model_invents_gives_up_or_fails() -> None:
    bad = GOOD.replace("100m", "999m")
    for reply in (bad, GIVE_UP, RuntimeError("rate limited")):
        out = asyncio.run(refine_tables(Scripted(reply), "m", [table()]))
        assert out[0].text == FLAT


def test_split_tables_are_not_rebuilt() -> None:
    parts = [table(part=1, parts=2), table(part=2, parts=2)]
    out = asyncio.run(refine_tables(Scripted(GOOD), "m", parts))
    assert [c.text for c in out] == [FLAT, FLAT]
