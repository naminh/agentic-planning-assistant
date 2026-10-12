from pathlib import Path

from kreuzberg import (  # type: ignore[attr-defined]
    ExtractionConfig,
    ImageExtractionConfig,
    PageConfig,
    PdfConfig,
    ResultFormat,
    extract_file,
)

from parser.document import PdfDocument

DEFAULT_IMAGE_DPI = 300


def _default_extraction_config() -> ExtractionConfig:
    """Build the default PDF extraction config for block-level parsing.

    Returns:
        ExtractionConfig with element-based format, page extraction, and images at 300 DPI.
    """
    return ExtractionConfig(
        result_format=ResultFormat.ELEMENT_BASED,
        pages=PageConfig(extract_pages=True),
        images=ImageExtractionConfig(
            extract_images=True,
            target_dpi=DEFAULT_IMAGE_DPI,
        ),
        pdf_options=PdfConfig(
            extract_metadata=True,
        ),
    )


async def extract_pdf_document(
    file_path: str | Path,
    config: ExtractionConfig | None = None,
) -> PdfDocument:
    """Extract a PDF into document, page, text-block, and image-block schemas.

    Args:
        file_path: Path to the source PDF file.
        config: Optional kreuzberg extraction config override. If None, uses default config.

    Returns:
        PdfDocument containing extracted content and metadata.
    """
    path = Path(file_path).expanduser().resolve()
    extraction_config = config or _default_extraction_config()
    result = await extract_file(path, config=extraction_config)
    return PdfDocument.from_extraction_result(path, result)
