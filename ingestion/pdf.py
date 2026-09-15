from pathlib import Path

import fitz

from app.models import DocumentPage


def extract_pages(
    pdf_path: Path,
    document_id: str,
) -> list[DocumentPage]:
    """Extract page-level text from a PDF."""
    document = fitz.open(pdf_path)

    pages: list[DocumentPage] = []

    for page_index, page in enumerate(document):
        text = page.get_text("text").strip()

        pages.append(
            DocumentPage(
                document_id=document_id,
                page_number=page_index + 1,
                text=text,
            )
        )

    document.close()

    return pages