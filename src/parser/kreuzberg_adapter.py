"""Adapter for Kreuzberg extraction results.

Kreuzberg extracts text, images, and metadata from PDFs.
"""

from collections.abc import Mapping, Sequence
from typing import Any


class KreuzbergResult:
    """Wrapper for Kreuzberg extraction results with typed accessors."""

    def __init__(self, raw_result: Any):
        """Initialize from a raw Kreuzberg extraction result.

        Args:
            raw_result: The extraction result object returned by Kreuzberg.
        """
        self._result = raw_result

    @property
    def mime_type(self) -> str | None:
        """Return the detected MIME type."""
        return getattr(self._result, "mime_type", None)

    @property
    def elements(self) -> Sequence[Mapping[str, Any]]:
        """Return raw extraction elements."""
        return getattr(self._result, "elements", []) or []

    @property
    def pages(self) -> Sequence[Mapping[str, Any]]:
        """Return raw page payloads."""
        return getattr(self._result, "pages", []) or []

    @property
    def images(self) -> Sequence[Mapping[str, Any]]:
        """Return raw image payloads."""
        return getattr(self._result, "images", []) or []

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return document-level metadata."""
        raw_metadata = getattr(self._result, "metadata", None)
        return raw_metadata if isinstance(raw_metadata, Mapping) else {}

    def iter_text_elements(self) -> list[Mapping[str, Any]]:
        """Return extractor elements that contain user-facing text.

        Filters out page breaks, images, and empty text elements.
        Returns elements sorted by their original extraction order.
        """
        text_elements = []
        for element in self.elements:
            if self._is_page_break_element(element) or self._is_image_element(element):
                continue

            text = str(element.get("text") or "").strip()
            if not text:
                continue
            text_elements.append(element)

        return sorted(text_elements, key=self._element_sort_key)

    def iter_page_images(self) -> list[Mapping[str, Any]]:
        """Return the best available page-scoped image list.

        Prefers images extracted from pages over document-level images.
        """
        images_from_pages = [
            image for raw_page in self.pages for image in raw_page.get("images", [])
        ]
        return images_from_pages or list(self.images)

    def page_dimensions(self) -> dict[int, tuple[float | None, float | None]]:
        """Extract page dimensions from document metadata when present.

        Returns:
            Mapping of 1-based page number to (width, height) tuples.
        """
        pages_metadata = self._get_mapping(
            self._get_mapping(self.metadata).get("pages")
        ).get("pages", [])
        dimensions: dict[int, tuple[float | None, float | None]] = {}

        for page_metadata in pages_metadata:
            page_number = self.extract_page_number(page_metadata)
            if page_number is None:
                continue

            raw_dimensions = list(page_metadata.get("dimensions") or [])
            width = (
                self._optional_float(raw_dimensions[0])
                if len(raw_dimensions) >= 1
                else None
            )
            height = (
                self._optional_float(raw_dimensions[1])
                if len(raw_dimensions) >= 2
                else None
            )
            dimensions[page_number] = (width, height)

        return dimensions

    def page_count(self) -> int:
        """Read a page count from document metadata."""
        raw_page_count = self.metadata.get("page_count")
        if raw_page_count is None:
            raw_page_count = self._get_mapping(self.metadata.get("pages")).get(
                "total_count", 0
            )
        return int(raw_page_count or 0)

    @staticmethod
    def extract_page_number(payload: Mapping[str, Any] | None) -> int | None:
        """Extract a 1-based page number from a metadata payload.

        Args:
            payload: Metadata payload that may contain a page number field.
        """
        if not isinstance(payload, Mapping):
            return None
        raw_page_number = payload.get("page_number") or payload.get("number")
        if raw_page_number is None:
            return None
        return int(raw_page_number)

    @staticmethod
    def _is_page_break_element(element: Mapping[str, Any]) -> bool:
        """Identify Kreuzberg's synthetic page-break marker."""
        return str(element.get("element_type") or "") == "page_break"

    @staticmethod
    def _is_image_element(element: Mapping[str, Any]) -> bool:
        """Identify image elements that should become image blocks, not text blocks."""
        return str(element.get("element_type") or "") == "image"

    @staticmethod
    def _element_sort_key(element: Mapping[str, Any]) -> tuple[int, str]:
        """Preserve extractor ordering where available."""
        metadata = element.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        raw_index = metadata.get("element_index")
        return (int(raw_index or 0), str(element.get("element_id") or ""))

    @staticmethod
    def _get_mapping(value: Any) -> Mapping[str, Any]:
        """Return a mapping-like value or an empty mapping."""
        if isinstance(value, Mapping):
            return value
        return {}

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        """Convert numeric metadata values to floats."""
        return float(value) if value is not None else None
