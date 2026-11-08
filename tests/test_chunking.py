from app.models import DocumentPage
from chunking.chunker import chunk_pages


def test_chunk_pages_preserves_page_metadata() -> None:
    pages = [
        DocumentPage(
            document_id="contract-1",
            page_number=3,
            text="First paragraph.\n\nSecond paragraph.",
            extraction_method="text",
        )
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "contract-1"
    assert chunks[0]["page_number"] == 3
    assert "First paragraph." in chunks[0]["text"]
    assert "Second paragraph." in chunks[0]["text"]