"""Page-aware text extraction for PDFs with an extractable text layer."""

import re
from dataclasses import dataclass

import pymupdf

from app.ingestion.errors import IngestionError

MIN_MEANINGFUL_CHARACTERS = 20
PAGE_SEPARATOR = "\n\n\f\n\n"


@dataclass(frozen=True)
class PdfPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedPdf:
    pages: tuple[PdfPage, ...]
    text: str


def normalize_page_text(text: str) -> str:
    normalized_lines: list[str] = []
    blank_pending = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t\v\f ]+", " ", raw_line).strip()
        if not line:
            if normalized_lines:
                blank_pending = True
            continue
        if blank_pending:
            normalized_lines.append("")
        normalized_lines.append(line)
        blank_pending = False
    return "\n".join(normalized_lines).strip()


def extract_pdf(content: bytes) -> ExtractedPdf:
    if not content.startswith(b"%PDF-"):
        raise IngestionError("PDF response has an invalid %PDF- signature")
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise IngestionError(f"could not open PDF: {exc}") from exc

    try:
        if document.needs_pass:
            raise IngestionError("PDF is encrypted or password-protected")
        if document.page_count == 0:
            raise IngestionError("PDF contains no pages")

        pages = tuple(
            PdfPage(
                page_number=page_number,
                text=normalize_page_text(page.get_text("text", sort=True)),
            )
            for page_number, page in enumerate(document, start=1)
        )
    except IngestionError:
        raise
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise IngestionError(f"could not extract PDF text: {exc}") from exc
    finally:
        document.close()

    text = PAGE_SEPARATOR.join(page.text for page in pages if page.text)
    meaningful_characters = sum(character.isalnum() for character in text)
    if meaningful_characters < MIN_MEANINGFUL_CHARACTERS:
        raise IngestionError("PDF contains no meaningful extractable text; OCR is required")
    return ExtractedPdf(pages=pages, text=text)
