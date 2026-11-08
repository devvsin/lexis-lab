import re

from app.models import DocumentPage


def chunk_pages(
    pages: list[DocumentPage],
    max_chars: int = 2000,
) -> list[dict]:
    chunks = []

    for page in pages:
        paragraphs = re.split(r"\n\s*\n", page.text)

        current_text = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if current_text and len(current_text) + len(paragraph) + 1 > max_chars:
                chunks.append(
                    {
                        "chunk_id": f"{page.document_id}-{page.page_number}-{len(chunks)}",
                        "document_id": page.document_id,
                        "page_number": page.page_number,
                        "text": current_text.strip(),
                    }
                )
                current_text = ""

            current_text += f"{paragraph}\n"

        if current_text.strip():
            chunks.append(
                {
                    "chunk_id": f"{page.document_id}-{page.page_number}-{len(chunks)}",
                    "document_id": page.document_id,
                    "page_number": page.page_number,
                    "text": current_text.strip(),
                }
            )

    return chunks