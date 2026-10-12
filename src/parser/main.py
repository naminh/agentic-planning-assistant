"""Simple Typer app to parse PDFs into structured JSON output."""

import asyncio
from pathlib import Path

import typer
from loguru import logger

from parser.document import PdfDocument
from parser.parser import extract_pdf_document

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "extracted"


app = typer.Typer(
    name="parser",
    help="Extract PDFs into structured JSON output.",
    add_completion=True,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    input_dir: Path = typer.Option(
        DEFAULT_DATA_DIR,
        help="Folder to recursively scan for PDF files.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Folder where extracted JSON files will be written.",
    ),
    max_concurrency: int = typer.Option(
        16,
        min=1,
        help="Maximum number of PDFs to extract concurrently.",
    ),
) -> None:
    """Extract all PDFs under a folder into mirrored JSON output.

    Args:
        input_dir: Folder to recursively scan for PDF files.
        output_dir: Folder where extracted JSON files will be written.
        max_concurrency: Maximum number of PDFs to extract at once.
    """
    asyncio.run(
        _extract_pdfs(
            input_dir=input_dir,
            output_dir=output_dir,
            max_concurrency=max_concurrency,
        )
    )


async def _extract_pdfs(
    input_dir: Path,
    output_dir: Path,
    max_concurrency: int,
) -> None:
    """Extract all PDFs from an input directory tree.

    Args:
        input_dir: Folder to recursively scan for PDF files.
        output_dir: Folder where extracted JSON files will be written.
        max_concurrency: Maximum number of PDFs to extract at once.
    """
    resolved_input_dir = input_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(resolved_input_dir.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {resolved_input_dir}")
        return

    semaphore = asyncio.Semaphore(max_concurrency)
    await asyncio.gather(
        *[
            _extract_single_pdf(
                pdf_path=pdf_path,
                input_dir=resolved_input_dir,
                output_dir=resolved_output_dir,
                semaphore=semaphore,
            )
            for pdf_path in pdf_paths
        ]
    )


async def _extract_single_pdf(
    pdf_path: Path,
    input_dir: Path,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """Extract one PDF and write mirrored JSON output.

    Args:
        pdf_path: Source PDF file to extract.
        input_dir: Root input directory used to preserve folder structure.
        output_dir: Root output directory for extracted JSON files.
        semaphore: Concurrency limiter shared across extraction tasks.
    """
    async with semaphore:
        document = await extract_pdf_document(pdf_path)
        _write_document_json(
            document=document,
            source_path=pdf_path,
            input_dir=input_dir,
            output_dir=output_dir,
        )


def _write_document_json(
    document: PdfDocument,
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Write an extracted document to mirrored JSON output.

    Args:
        document: Extracted PDF document to persist.
        source_path: Original PDF path relative to the input directory tree.
        input_dir: Root input directory used to preserve folder structure.
        output_dir: Root output directory for extracted JSON files.
    """
    relative_path = source_path.relative_to(input_dir).with_suffix(".json")
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.to_json(), encoding="utf-8")
    logger.info(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
