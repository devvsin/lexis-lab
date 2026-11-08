from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """Semantic retrieval using sentence embeddings."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embeddings = self.model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Return the top-k semantically similar chunks."""

        if not query.strip() or not self.chunks or top_k <= 0:
            return []

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        scores = self.embeddings @ query_embedding

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            {
                **self.chunks[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices[:top_k]
        ]