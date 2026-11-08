
from pathlib import Path

from chunking.chunker import chunk_pages
from evaluation.dataset import EVALUATION_DATASET
from ingestion.pdf import extract_pages
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.semantic import SemanticRetriever

PDF_PATH = Path("data/termination-settlement-agreement.pdf")
DOCUMENT_ID = "termination-settlement"


def main() -> None:
    pages = extract_pages(PDF_PATH, DOCUMENT_ID)
    chunks = chunk_pages(pages)

    retrievers = {
        "BM25": BM25Retriever(chunks),
        "Semantic": SemanticRetriever(chunks),
        "Hybrid": HybridRetriever(chunks),
    }

    for item in EVALUATION_DATASET:
        print("\n" + "=" * 70)
        print(f"QUESTION: {item['question']}")
        print(f"ANSWERABLE: {item['answerable']}")
        print(f"EXPECTED CHUNKS: {item['relevant_chunk_ids']}")

        for name, retriever in retrievers.items():
            results = retriever.search(
                item["question"],
                top_k=3,
            )

            retrieved_ids = [
                result["chunk_id"] for result in results
            ]

            relevant_found = any(
                chunk_id in item["relevant_chunk_ids"]
                for chunk_id in retrieved_ids
            )

            print(f"\n{name}")
            print(f"Retrieved IDs: {retrieved_ids}")

            if item["answerable"]:
                print(f"Relevant chunk found: {relevant_found}")

            for result in results:
                print(
                    f"  Page {result['page_number']} | "
                    f"Score: {result['score']:.4f}"
                )
                print(f"  {result['text'][:180]}...")


if __name__ == "__main__":
    main()
