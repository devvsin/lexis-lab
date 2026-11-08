from pathlib import Path

from chunking.chunker import chunk_pages
from ingestion.pdf import extract_pages
from retrieval.bm25 import BM25Retriever


class LegalRetriever:
    def __init__(self, pdf_path: Path, document_id: str) -> None:
        pages = extract_pages(pdf_path, document_id)
        chunks = chunk_pages(pages)

        self.retriever = BM25Retriever(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        return self.retriever.search(query, top_k=top_k)