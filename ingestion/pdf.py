
from pathlib import Path

import fitz

from app.models import DocumentPage


def ocr_page(page: fitz.Page) -> str:
    """Import the OCR engine only when OCR is actually required."""
    from ingestion.ocr import ocr_page as run_ocr

    return run_ocr(page)


def extract_pages(pdf_path: Path, document_id: str) -> list[DocumentPage]:
    document = fitz.open(pdf_path)
    pages = []

    try:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            if text:
                extraction_method = "text"
            else:
                text = ocr_page(page)
                extraction_method = "ocr" if text else "empty"

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    page_number=page_index + 1,
                    text=text,
                    extraction_method=extraction_method,
                )
            )
    finally:
        document.close()

    return pages
