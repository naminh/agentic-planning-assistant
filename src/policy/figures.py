"""Caption the figures, maps and diagrams in policy documents so they can be searched.

Policy PDFs carry meaning in images: a settlement diagram, a flood-zone map, a worked
example of a distance standard. Text extraction drops them. For sources marked
`captions: true`, each large embedded image is described once by a vision model; the
description is indexed as a `figure` chunk that points at a saved copy of the image, and
the assessor is shown the image itself whenever such a chunk is retrieved.
"""

import asyncio
import base64
import hashlib
import json
from pathlib import Path

from loguru import logger

from config import Settings
from domain import PolicyChunk
from llm import LLMClient
from policy.config import PolicySource

MIN_EDGE_PX = 300  # smaller images are icons, logos and rules
MAX_FIGURES_PER_SOURCE = 40
SKIP = "DECORATIVE"

FIGURE_SYSTEM = (
    "You are describing one figure from a planning policy document so that an officer "
    "can find it by searching. State what kind of figure it is (map, diagram, chart, "
    "photograph, worked example) and what it shows that bears on planning: places, "
    "zones, boundaries, distances, thresholds, labels and legends, quoting any written "
    "text or numbers exactly. Do not guess what is illegible. Be factual and under 150 "
    f"words. If it is only a logo, border, icon or decoration, reply exactly {SKIP}."
)


def figure_blocks(doc: dict[str, list[dict[str, object]]]) -> list[tuple[int, bytes]]:
    """(page, image bytes) for every large embedded image, in page order."""
    found = []
    for page in doc["pages"]:
        for block in page["blocks"]:  # type: ignore[attr-defined]
            if block.get("block_type") != "image":
                continue
            if min(block.get("width", 0), block.get("height", 0)) < MIN_EDGE_PX:
                continue
            raw = block.get("image_bytes")
            if isinstance(raw, str):
                found.append((int(page["page_number"]), base64.b64decode(raw)))  # type: ignore[call-overload]
    return found


def _save(directory: Path, data: bytes) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{hashlib.sha256(data).hexdigest()[:16]}.img"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


async def _caption(
    client: LLMClient,
    model: str,
    source: PolicySource,
    page: int,
    data: bytes,
    directory: Path,
    reasoning_off: bool = False,
) -> PolicyChunk | None:
    try:
        reply = await client.complete(
            model,
            FIGURE_SYSTEM,
            f"Source: {source.label}, page {page}.",
            images=[data],
            max_tokens=2500,  # reasoning models spend part of this thinking
            stage=f"figure:{source.stem}:{page}",
            reasoning_off=reasoning_off,
        )
    except Exception as err:  # a missing caption only means a figure is not searchable
        logger.warning(
            f"Figure p.{page} of {source.label} skipped: {type(err).__name__}"
        )
        return None
    caption = " ".join(reply.split())
    if not caption or caption.upper().startswith(SKIP):
        return None
    return PolicyChunk(
        source=source.label,
        page=page,
        text=f"Figure on page {page} of {source.label}: {caption}",
        version=source.version,
        kind="figure",
        heading=f"Figure p.{page}",
        image=_save(directory, data),
    )


async def caption_figures(
    settings: Settings, client: LLMClient, sources: tuple[PolicySource, ...]
) -> list[PolicyChunk]:
    """Figure chunks for every source marked `captions: true` (cached per image)."""
    directory = settings.cache_dir / "figures"
    tasks = []
    for source in sources:
        path = settings.extracted_dir / source.path
        if not source.captions or not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for page, data in figure_blocks(doc)[:MAX_FIGURES_PER_SOURCE]:
            tasks.append(
                _caption(
                    client,
                    settings.figure_model,
                    source,
                    page,
                    data,
                    directory,
                    settings.caption_reasoning_off,
                )
            )
    chunks = [c for c in await asyncio.gather(*tasks) if c is not None]
    logger.info(f"Figures captioned: {len(chunks)} of {len(tasks)}")
    return chunks


def figure_images(chunks: list[PolicyChunk], limit: int = 2) -> list[bytes]:
    """Image bytes for the first `limit` retrieved figure chunks that still exist."""
    images = []
    for chunk in chunks:
        if chunk.kind == "figure" and chunk.image and Path(chunk.image).exists():
            images.append(Path(chunk.image).read_bytes())
        if len(images) >= limit:
            break
    return images
