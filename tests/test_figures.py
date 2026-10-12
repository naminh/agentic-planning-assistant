"""Offline tests for figure captioning and for showing retrieved figures to the model."""

import asyncio
import base64
import json
from pathlib import Path

from config import Settings
from domain import PolicyChunk
from llm import LLMClient
from policy.config import PolicySource
from policy.figures import SKIP, caption_figures, figure_blocks, figure_images


def image(width: int, height: int, payload: bytes) -> dict:
    return {
        "block_type": "image",
        "width": width,
        "height": height,
        "image_bytes": base64.b64encode(payload).decode(),
    }


DOC = {
    "pages": [
        {"page_number": 4, "blocks": [image(900, 700, b"map"), image(80, 80, b"logo")]},
        {"page_number": 5, "blocks": [{"block_type": "text", "text": "x"}]},
        {"page_number": 6, "blocks": [image(600, 600, b"banner")]},
    ]
}


class Captioner(LLMClient):
    def __init__(self, replies: dict[bytes, str]) -> None:
        self.replies = replies
        self.seen_images: list[list[bytes]] = []

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
        self.seen_images.append(images or [])
        return self.replies[(images or [b""])[0]]


def settings(tmp_path: Path) -> Settings:
    return Settings(
        cheap_model="c",
        strong_model="s",
        judge_model="j",
        api_key="k",
        base_url="u",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )


def test_only_large_images_are_candidates() -> None:
    assert figure_blocks(DOC) == [(4, b"map"), (6, b"banner")]


def test_captions_become_searchable_figure_chunks_with_a_saved_image(
    tmp_path: Path,
) -> None:
    s = settings(tmp_path)
    source = PolicySource(path="p.json", label="Flood SPD", captions=True)
    path = s.extracted_dir / "p.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(DOC))
    client = Captioner(
        {b"map": "Map of flood zones 2 and 3 around the river.", b"banner": SKIP}
    )
    chunks = asyncio.run(caption_figures(s, client, (source,)))
    (chunk,) = chunks
    assert chunk.kind == "figure" and chunk.page == 4 and "flood zones" in chunk.text
    assert chunk.text.startswith("Figure on page 4 of Flood SPD")
    assert chunk.image and Path(chunk.image).read_bytes() == b"map"


def test_sources_without_the_captions_flag_are_not_touched(tmp_path: Path) -> None:
    s = settings(tmp_path)
    source = PolicySource(path="p.json", label="Plain")
    client = Captioner({})
    assert asyncio.run(caption_figures(s, client, (source,))) == []
    assert client.seen_images == []


def test_retrieved_figures_are_handed_to_the_assessor(tmp_path: Path) -> None:
    img = tmp_path / "f.img"
    img.write_bytes(b"pixels")
    figure = PolicyChunk(source="S", page=1, text="t", kind="figure", image=str(img))
    gone = PolicyChunk(
        source="S", page=2, text="t", kind="figure", image=str(tmp_path / "x")
    )
    prose = PolicyChunk(source="S", page=3, text="t")
    assert figure_images([prose, gone, figure]) == [b"pixels"]
