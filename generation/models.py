from pydantic import BaseModel


class Source(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    text: str


class GroundedAnswer(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float | None = None
    insufficient_evidence: bool
    citations_valid: bool