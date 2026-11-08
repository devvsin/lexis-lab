
import json
from pathlib import Path

from chunking.chunker import chunk_pages
from evaluation.runner import evaluate_all
from ingestion.pdf import extract_pages

PDF_PATH = Path("data/termination-settlement-agreement.pdf")
DOCUMENT_ID = "termination-settlement"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    pages = extract_pages(PDF_PATH, DOCUMENT_ID)
    chunks = chunk_pages(pages)

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print("\nRunning retrieval experiments...")

    results = evaluate_all(chunks, k=3)

    print("\n=== RETRIEVAL EXPERIMENT RESULTS ===")
    print(json.dumps(results, indent=2))

    output_path = Path("evaluation/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
