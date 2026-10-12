import base64
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator

from .kreuzberg_adapter import KreuzbergResult


class PdfBlockType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class PdfBlockBase(BaseModel):
    """Base schema shared by all extracted page blocks."""

    block_id: str = Field(..., description="Stable identifier for the extracted block")
    block_index: int = Field(
        ..., description="0-based index of the block within the page"
    )
    block_type: PdfBlockType = Field(..., description="High-level block type")
    page_number: int = Field(..., description="1-based page number")


class PdfTextBlock(PdfBlockBase):
    """A text-like block extracted from a PDF page."""

    block_type: Literal[PdfBlockType.TEXT] = PdfBlockType.TEXT
    text: str = Field(..., description="Extracted text content for the block")
    source_type: str = Field(
        ...,
        description="Original extractor element type for the block",
    )

    @classmethod
    def from_extracted_element(
        cls,
        element: Mapping[str, Any],
        *,
        block_index: int,
        page_number: int,
    ) -> "PdfTextBlock":
        """Build a text block from a raw extractor element.

        Args:
            element: Raw element payload returned by the PDF extraction backend.
            block_index: 0-based position of the block within the page.
            page_number: 1-based page number the block belongs to.
        """
        return cls(
            block_id=str(
                element.get("element_id") or f"text-{page_number}-{block_index}"
            ),
            block_index=block_index,
            page_number=page_number,
            text=str(element.get("text") or "").strip(),
            source_type=str(element.get("element_type") or "text"),
        )


class PdfImageBlock(PdfBlockBase):
    """An embedded image extracted from a PDF page."""

    block_type: Literal[PdfBlockType.IMAGE] = PdfBlockType.IMAGE
    image_bytes: bytes = Field(..., description="Raw extracted image bytes")
    image_format: str = Field(..., description="Image format, for example png or jpeg")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    description: str | None = Field(
        default=None,
        description="Optional backend-provided description for the image block",
    )

    @field_validator("image_bytes", mode="before")
    @classmethod
    def decode_base64_image_bytes(cls, value: Any) -> Any:
        """Decode base64 string to bytes on init."""
        if isinstance(value, str):
            return base64.b64decode(value)
        return value

    @field_serializer("image_bytes")
    def serialize_image_bytes(self, value: bytes) -> str:
        """Serialize image bytes to base64 string for JSON output."""
        return base64.b64encode(value).decode("ascii")

    @classmethod
    def from_extracted_image(
        cls,
        image: Mapping[str, Any],
        *,
        block_index: int,
        page_number: int,
    ) -> "PdfImageBlock | None":
        """Build an image block from a raw extractor image payload.

        Args:
            image: Raw image payload returned by the PDF extraction backend.
            block_index: 0-based position of the block within the page.
            page_number: 1-based page number the block belongs to.
        """
        image_bytes = image.get("data")
        image_format = image.get("format")
        if not isinstance(image_bytes, bytes) or image_format is None:
            return None

        return cls(
            block_id=f"image-{page_number}-{block_index}",
            block_index=block_index,
            page_number=page_number,
            image_bytes=image_bytes,
            image_format=str(image_format),
            width=int(image.get("width") or 0),
            height=int(image.get("height") or 0),
            description=(
                str(image.get("description"))
                if image.get("description") is not None
                else None
            ),
        )


type PdfBlock = Annotated[
    PdfTextBlock | PdfImageBlock,
    Field(discriminator="block_type"),
]


class PdfPage(BaseModel):
    """Structured content for a single PDF page."""

    page_number: int = Field(..., description="1-based page number")
    width: float | None = Field(
        default=None, description="Page width in PDF points when available"
    )
    height: float | None = Field(
        default=None, description="Page height in PDF points when available"
    )
    is_blank: bool = Field(
        default=False, description="Whether the extractor marked the page blank"
    )
    blocks: list[PdfBlock] = Field(
        default_factory=list,
        description="Text and image blocks extracted from the page",
    )

    @classmethod
    def create(
        cls,
        *,
        page_number: int,
        width: float | None = None,
        height: float | None = None,
        is_blank: bool = False,
    ) -> "PdfPage":
        """Create a page shell ready to receive blocks.

        Args:
            page_number: 1-based page number.
            width: Page width in PDF points, when available.
            height: Page height in PDF points, when available.
            is_blank: Whether the extractor marked the page as blank.
        """
        return cls(
            page_number=page_number,
            width=width,
            height=height,
            is_blank=is_blank,
        )

    def add_text_block(self, element: Mapping[str, Any]) -> None:
        """Append a text block built from a raw extractor element.

        Args:
            element: Raw text-like element returned by the PDF extraction backend.
        """
        self.blocks.append(
            PdfTextBlock.from_extracted_element(
                element,
                block_index=len(self.blocks),
                page_number=self.page_number,
            )
        )

    def add_image_block(self, image: Mapping[str, Any]) -> None:
        """Append an image block built from a raw extractor image payload.

        Args:
            image: Raw image payload returned by the PDF extraction backend.
        """
        block = PdfImageBlock.from_extracted_image(
            image,
            block_index=len(self.blocks),
            page_number=self.page_number,
        )
        if block is not None:
            self.blocks.append(block)

    @property
    def text_blocks(self) -> list[PdfTextBlock]:
        """Return only text blocks from the page."""
        return [block for block in self.blocks if isinstance(block, PdfTextBlock)]

    @property
    def image_blocks(self) -> list[PdfImageBlock]:
        """Return only image blocks from the page."""
        return [block for block in self.blocks if isinstance(block, PdfImageBlock)]

    @property
    def text(self) -> str:
        """Return the concatenated page text."""
        return "".join(block.text for block in self.text_blocks)


class PdfDocument(BaseModel):
    """Structured representation of a parsed PDF document."""

    source_path: str = Field(..., description="Absolute path to the PDF file")
    filename: str = Field(..., description="Filename of the source PDF")
    mime_type: str | None = Field(default=None, description="Detected mime type")
    page_count: int = Field(..., description="Number of pages in the document")
    pages: list[PdfPage] = Field(
        default_factory=list,
        description="Structured page content extracted from the PDF",
    )

    @property
    def path(self) -> Path:
        """Return the document path as a Path object."""
        return Path(self.source_path)

    @property
    def text_blocks(self) -> list[PdfTextBlock]:
        """Return every text block in the document."""
        return [block for page in self.pages for block in page.text_blocks]

    @property
    def image_blocks(self) -> list[PdfImageBlock]:
        """Return every image block in the document."""
        return [block for page in self.pages for block in page.image_blocks]

    @property
    def text(self) -> str:
        """Return the concatenated document text."""
        return "".join(block.text for block in self.text_blocks)

    def get_texts(self) -> list[str]:
        """Return extracted text block contents in document order."""
        return [block.text for block in self.text_blocks]

    def get_images(self) -> list[bytes]:
        """Return extracted image bytes in document order."""
        return [block.image_bytes for block in self.image_blocks]

    def to_json(self) -> str:
        """Return the document as a JSON string."""
        import json

        return json.dumps(
            self.model_dump(),
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def _build_pages(cls, kr: KreuzbergResult) -> dict[int, PdfPage]:
        """Build page shells before blocks are attached.

        Args:
            kr: Kreuzberg result adapter with typed accessors.
        """
        page_dimensions = kr.page_dimensions()
        default_width = kr._optional_float(kr.metadata.get("width"))
        default_height = kr._optional_float(kr.metadata.get("height"))

        pages_by_number: dict[int, PdfPage] = {}
        for raw_page in kr.pages:
            page_number = int(raw_page["page_number"])
            width, height = page_dimensions.get(
                page_number, (default_width, default_height)
            )
            pages_by_number[page_number] = PdfPage.create(
                page_number=page_number,
                width=width,
                height=height,
                is_blank=bool(raw_page.get("is_blank", False)),
            )

        if pages_by_number:
            return pages_by_number

        page_count = kr.page_count()
        for page_number in range(1, page_count + 1):
            width, height = page_dimensions.get(
                page_number, (default_width, default_height)
            )
            pages_by_number[page_number] = PdfPage.create(
                page_number=page_number,
                width=width,
                height=height,
            )
        return pages_by_number

    @staticmethod
    def _get_or_create_page(
        pages_by_number: dict[int, PdfPage], page_number: int
    ) -> PdfPage:
        """Return an existing page or create a blank shell on demand.

        Args:
            pages_by_number: Mutable mapping of page number to page object.
            page_number: 1-based page number to look up or create.
        """
        page = pages_by_number.get(page_number)
        if page is None:
            page = PdfPage.create(page_number=page_number)
            pages_by_number[page_number] = page
        return page

    @classmethod
    def from_extraction_result(
        cls,
        source_path: str | Path,
        result: Any,
    ) -> "PdfDocument":
        """Build a document from a kreuzberg extraction result.

        Args:
            source_path: Path to the original PDF file.
            result: Extraction result object returned by `kreuzberg`.
        """
        path = Path(source_path).expanduser().resolve()
        kr = KreuzbergResult(result)
        pages_by_number = cls._build_pages(kr)

        for element in kr.iter_text_elements():
            page_number = kr.extract_page_number(element.get("metadata"))
            if page_number is None:
                continue
            cls._get_or_create_page(pages_by_number, page_number).add_text_block(
                element
            )

        for image in kr.iter_page_images():
            page_number = kr.extract_page_number(image)
            if page_number is None:
                continue
            cls._get_or_create_page(pages_by_number, page_number).add_image_block(image)

        pages = [
            pages_by_number[page_number] for page_number in sorted(pages_by_number)
        ]
        return cls(
            source_path=str(path),
            filename=path.name,
            mime_type=kr.mime_type,
            page_count=len(pages),
            pages=pages,
        )
