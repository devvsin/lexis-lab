
from pathlib import Path

import fitz

from generation.models import GroundedAnswer
from generation.pipeline import LegalQAPipeline


def create_test_pdf(
    pdf_path: Path,
    text: str,
) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(pdf_path)
    document.close()


def test_pipeline_retrieves_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "contract.pdf"
    create_test_pdf(
        pdf_path,
        "The agreement may be terminated with thirty days notice.",
    )

    def fake_answer(question, sources):
        return GroundedAnswer(
            answer="The agreement may be terminated with thirty days notice. [Page 1]",
            sources=sources,
            confidence=0.9,
            insufficient_evidence=False,
            citations_valid=True,
        )

    monkeypatch.setattr(
        "generation.pipeline.generate_answer",
        fake_answer,
    )

    pipeline = LegalQAPipeline(pdf_path, "contract-001")
    result = pipeline.answer(
        "How can the agreement be ended?",
        top_k=1,
    )

    assert "thirty days notice" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].page_number == 1
    assert result.citations_valid is True
    assert result.insufficient_evidence is False


def test_pipeline_abstains_when_no_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "contract.pdf"
    create_test_pdf(
        pdf_path,
        "The agreement may be terminated with thirty days notice.",
    )

    def fake_answer(question, sources):
        assert sources == []

        return GroundedAnswer(
            answer=(
                "I don't have enough evidence in the document "
                "to answer this question."
            ),
            sources=[],
            confidence=0.0,
            insufficient_evidence=True,
            citations_valid=False,
        )

    monkeypatch.setattr(
        "generation.pipeline.HybridRetriever.search",
        lambda self, question, top_k=3: [],
    )

    pipeline = LegalQAPipeline(pdf_path, "contract-001")
    result = pipeline.answer(
        "What is the company's office address on Mars?",
        top_k=1,
    )

    assert result.insufficient_evidence is True
    assert result.sources == []
    assert result.confidence == 0.0
    assert "sufficient evidence" in result.answer.lower()
