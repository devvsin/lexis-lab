from pathlib import Path

import fitz

from retrieval.pipeline import LegalRetriever


def test_legal_retriever_searches_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "contract.pdf"

    document = fitz.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        "The agreement may be terminated with thirty days notice.",
    )

    document.save(pdf_path)
    document.close()

    retriever = LegalRetriever(pdf_path, "contract-001")

    results = retriever.search("termination notice", top_k=1)

    assert len(results) == 1
    assert results[0]["document_id"] == "contract-001"
    assert results[0]["page_number"] == 1
    assert "terminated" in results[0]["text"]