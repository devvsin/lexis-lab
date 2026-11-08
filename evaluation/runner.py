from evaluation.dataset import EVALUATION_DATASET
from evaluation.retrieval import recall_at_k
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.semantic import SemanticRetriever


def evaluate_retriever(
    retriever,
    dataset: list[dict],
    k: int = 3,
) -> dict:
    """Evaluate retrieval performance on answerable and unanswerable queries."""

    answerable_scores = []
    unanswerable_hits = 0
    answerable_count = 0
    unanswerable_count = 0

    for item in dataset:
        results = retriever.search(
            item["question"],
            top_k=k,
        )

        if item["answerable"]:
            answerable_count += 1

            score = recall_at_k(
                results,
                item["relevant_chunk_ids"],
                k,
            )

            answerable_scores.append(score)

        else:
            unanswerable_count += 1

            if results:
                unanswerable_hits += 1

    recall = (
        sum(answerable_scores) / len(answerable_scores)
        if answerable_scores
        else 0.0
    )

    return {
        "recall_at_k": recall,
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "unanswerable_retrieval_rate": (
            unanswerable_hits / unanswerable_count
            if unanswerable_count
            else 0.0
        ),
    }


def evaluate_all(
    chunks: list[dict],
    k: int = 3,
) -> dict:
    """Compare all retrieval strategies."""

    retrievers = {
        "bm25": BM25Retriever(chunks),
        "semantic": SemanticRetriever(chunks),
        "hybrid": HybridRetriever(chunks),
    }

    return {
        name: evaluate_retriever(
            retriever,
            EVALUATION_DATASET,
            k=k,
        )
        for name, retriever in retrievers.items()
    }