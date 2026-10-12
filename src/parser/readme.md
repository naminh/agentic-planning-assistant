# PDF parser

This package contains:

- `parser.py`: asynchronous PDF extraction entry point
- `document.py`: document, page, text-block, and image-block models
- `main.py`: Typer CLI for batch extraction

## CLI

Run:

```bash
uv run parser
```

The command recursively scans `data` for PDFs and writes extracted JSON under `data/extracted`, preserving the input directory structure.

## Core API

Most consumers only need `extract_pdf_document` and the document helper methods:

```python
from parser.parser import extract_pdf_document

doc = await extract_pdf_document("/path/to/file.pdf")
texts = doc.get_texts()  # list[str]
images = doc.get_images()  # list[bytes]
```

`get_texts()` returns text blocks in document order.

`get_images()` returns raw image bytes in document order. These bytes can be passed to multimodal models or embedding workflows.

## Output schema

Each PDF is represented as a `PdfDocument` containing ordered pages and blocks. A shortened JSON example is shown below:

```json
{
  "source_path": "/workspace/data/local_policy/example.pdf",
  "filename": "example.pdf",
  "mime_type": "application/pdf",
  "page_count": 1,
  "pages": [
    {
      "page_number": 1,
      "width": 595.28,
      "height": 841.89,
      "is_blank": false,
      "blocks": [
        {
          "block_id": "text-1-0",
          "block_index": 0,
          "block_type": "text",
          "page_number": 1,
          "text": "Example extracted text.",
          "source_type": "narrative_text"
        }
      ]
    }
  ]
}
```
