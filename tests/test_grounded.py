from generation.grounded import validate_citations
from generation.models import Source


def test_valid_citation():
    sources = [
        Source(
            chunk_id="chunk-1",
            document_id="contract-001",
            page_number=3,
            text="The compensation is €400,000.",
        )
    ]

    assert validate_citations(
        "The compensation is €400,000. [Page 3]",
        sources,
    )


def test_rejects_unknown_page():
    sources = [
        Source(
            chunk_id="chunk-1",
            document_id="contract-001",
            page_number=3,
            text="The compensation is €400,000.",
        )
    ]

    assert not validate_citations(
        "The compensation is €400,000. [Page 9]",
        sources,
    )


def test_rejects_missing_citation():
    sources = [
        Source(
            chunk_id="chunk-1",
            document_id="contract-001",
            page_number=3,
            text="The compensation is €400,000.",
        )
    ]

    assert not validate_citations(
        "The compensation is €400,000.",
        sources,
    )


def test_rejects_citation_without_sources():
    assert not validate_citations(
        "The compensation is €400,000. [Page 3]",
        [],
    )
