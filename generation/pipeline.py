
import time
from pathlib import Path

from chunking.chunker import chunk_pages
from generation.grounded import build_context, insufficient_evidence
from generation.llm import generate_answer
from ingestion.pdf import extract_pages
from retrieval.hybrid import HybridRetriever


class LegalQAPipeline:
    """End-to-end legal document question-answering pipeline."""

    def __init__(self, pdf_path: Path, document_id: str) -> None:
        pages = extract_pages(pdf_path, document_id)
        chunks = chunk_pages(pages)

        self.retriever = HybridRetriever(chunks)

    def answer(self, question: str, top_k: int = 5):
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        results = self.retriever.search(question, top_k=top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        print(f"[TIMING] Retrieval: {retrieval_time:.2f}s")

        if insufficient_evidence(results):
            generation_start = time.perf_counter()
            answer = generate_answer(question, [])
            generation_time = time.perf_counter() - generation_start

            print(f"[TIMING] Generation: {generation_time:.2f}s")
            print(
                f"[TIMING] Total: "
                f"{time.perf_counter() - total_start:.2f}s"
            )
            return answer

        sources = build_context(results)

        generation_start = time.perf_counter()
        answer = generate_answer(question, sources)
        generation_time = time.perf_counter() - generation_start

        print(f"[TIMING] Generation: {generation_time:.2f}s")
        print(
            f"[TIMING] Total: "
            f"{time.perf_counter() - total_start:.2f}s"
        )

        return answer
