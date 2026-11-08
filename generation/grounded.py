import re

from generation.models import Source

    
def build_context(results: list[dict]) -> list[Source]:
    """Convert retrieval results into citation-ready sources."""

    return [
        Source(
            chunk_id=result["chunk_id"],
            document_id=result["document_id"],
            page_number=result["page_number"],
            text=result["text"],
        )
        for result in results
    ]


def insufficient_evidence(results: list[dict]) -> bool:
    """Determine whether retrieval found usable evidence."""

    return len(results) == 0


def validate_citations(answer: str, sources: list[Source]) -> bool:
    """Validate that every cited page exists in the supplied sources."""
    if not sources:
        return False

    cited_pages = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
    )

    # Answers with sources must include at least one citation.
    if not cited_pages:
        return False

    available_pages = {
        source.page_number for source in sources
    }

    # Every citation must refer to a page in the supplied sources.
    return all(
        int(page) in available_pages
        for page in cited_pages
    )
