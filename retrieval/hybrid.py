from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever


class HybridRetriever:
    """Combine BM25 and semantic retrieval."""

    def __init__(
        self,
        chunks: list[dict],
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        if bm25_weight < 0 or semantic_weight < 0:
            raise ValueError("Weights must be non-negative.")

        if bm25_weight + semantic_weight == 0:
            raise ValueError("At least one weight must be greater than zero.")

        self.bm25 = BM25Retriever(chunks)
        self.semantic = SemanticRetriever(chunks)

        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Return results using combined BM25 and semantic scores."""

        if not query.strip() or top_k <= 0:
            return []

        bm25_results = self.bm25.search(
            query,
            top_k=len(self.bm25.chunks),
        )

        semantic_results = self.semantic.search(
            query,
            top_k=len(self.semantic.chunks),
        )

        combined: dict[str, dict] = {}

        # Normalize BM25 scores.
        bm25_max = max(
            (result["score"] for result in bm25_results),
            default=0.0,
        )

        for result in bm25_results:
            chunk_id = result["chunk_id"]

            normalized_score = (
                result["score"] / bm25_max
                if bm25_max > 0
                else 0.0
            )

            combined[chunk_id] = {
                **result,
                "bm25_score": normalized_score,
                "semantic_score": 0.0,
            }

        # Add semantic scores.
        for result in semantic_results:
            chunk_id = result["chunk_id"]

            if chunk_id not in combined:
                combined[chunk_id] = {
                    **result,
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }

            combined[chunk_id]["semantic_score"] = result["score"]

        # Calculate combined score.
        for result in combined.values():
            result["score"] = (
                self.bm25_weight * result["bm25_score"]
                + self.semantic_weight * result["semantic_score"]
            )

        ranked_results = sorted(
            combined.values(),
            key=lambda result: result["score"],
            reverse=True,
        )

        return ranked_results[:top_k]