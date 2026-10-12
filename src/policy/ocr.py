"""Transcribe scanned (image-only) policy PDFs with a vision model, cached on disk."""

import asyncio
import json
from pathlib import Path

from loguru import logger

from config import Settings
from llm import LLMClient
from parser.document import PdfDocument
from policy.config import PolicySource

OCR_SYSTEM = (
    "You transcribe scanned planning policy documents for a search index. Output the "
    "page's text verbatim, preserving headings and numbered clauses. If the page "
    "contains a map, write 'MAP:' followed by what area it shows and any labels or "
    "street names that are legible. Do not summarise and do not add commentary."
)


def ocr_path(settings: Settings, source: PolicySource) -> Path:
    """Where the transcribed pages of a scanned source are stored."""
    return Path(settings.cache_dir / "policy_ocr" / f"{source.stem}.json")


def load_ocr_pages(settings: Settings, source: PolicySource) -> dict[int, str]:
    """Return page number -> transcript for a scanned source (empty if not yet run)."""
    path = ocr_path(settings, source)
    if not source.ocr or not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


async def transcribe_scanned_sources(
    settings: Settings, client: LLMClient, sources: tuple[PolicySource, ...]
) -> None:
    """OCR every source flagged `ocr=True` that has not been transcribed yet."""
    for source in (s for s in sources if s.ocr):
        target = ocr_path(settings, source)
        if target.exists():
            continue
        parsed = settings.extracted_dir / source.path
        if not parsed.exists():
            logger.warning(f"Cannot OCR {source.label}: run `uv run parser` first")
            continue
        pdf = PdfDocument.model_validate_json(parsed.read_text(encoding="utf-8"))
        logger.info(f"OCR: {source.label} ({pdf.page_count} pages)")

        async def one(page_number: int, image: bytes, label: str = source.label) -> str:
            reply = await client.complete(
                settings.strong_model,
                OCR_SYSTEM,
                f"Document: {label}, page {page_number}.",
                images=[image],
                max_tokens=3000,
                stage="policy_ocr",
            )
            return str(reply)

        pages = [(p.page_number, p.image_blocks) for p in pdf.pages if p.image_blocks]
        texts = await asyncio.gather(
            *(one(n, blocks[0].image_bytes) for n, blocks in pages)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {n: t for (n, _), t in zip(pages, texts, strict=True)}, indent=1
            ),
            encoding="utf-8",
        )
