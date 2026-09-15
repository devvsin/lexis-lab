import fitz

from ingestion.pdf import extract_pages


def test_extract_pages(tmp_path) -> None:
    pdf_path = tmp_path / "test.pdf"

    document = fitz.open()

    page = document.new_page()
    page.insert_text((72, 72), "This is page one.")

    page = document.new_page()
    page.insert_text((72, 72), "This is page two.")

    document.save(pdf_path)
    document.close()

    pages = extract_pages(pdf_path, "test-document")

    assert len(pages) == 2

    assert pages[0].document_id == "test-document"
    assert pages[0].page_number == 1
    assert "page one" in pages[0].text

    assert pages[1].page_number == 2
    assert "page two" in pages[1].text