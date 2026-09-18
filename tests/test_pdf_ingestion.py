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

    assert pages[0].extraction_method == "text"
    assert pages[1].extraction_method == "text"

def test_empty_page_is_detected(tmp_path) -> None:
    pdf_path = tmp_path / "empty.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    pages = extract_pages(pdf_path, "empty-document")

    assert len(pages) == 1
    assert pages[0].text == ""
    assert pages[0].extraction_method == "empty"    

def test_ocr_page_is_used_when_text_is_missing(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "scanned.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    monkeypatch.setattr(
        "ingestion.pdf.ocr_page",
        lambda page: "OCR extracted legal text.",
    )

    pages = extract_pages(pdf_path, "scanned-document")

    assert len(pages) == 1
    assert pages[0].text == "OCR extracted legal text."
    assert pages[0].extraction_method == "ocr"