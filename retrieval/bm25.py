import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Keyword-based retrieval using the BM25 ranking algorithm."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

        self.tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.index = (
            BM25Okapi(self.tokenized_chunks)
            if self.tokenized_chunks
            else None
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Return the top-k chunks relevant to a query."""

        if not query.strip() or not self.chunks or top_k <= 0:
            return []

        query_tokens = tokenize(query)

        if not query_tokens or self.index is None:
            return []

        # Check whether at least one query term
        # exists anywhere in the document collection.
        corpus_terms = {
            token
            for document in self.tokenized_chunks
            for token in document
        }

        if not any(token in corpus_terms for token in query_tokens):
            return []

        # Calculate BM25 relevance scores.
        scores = self.index.get_scores(query_tokens)

        # Rank chunks from highest score to lowest score.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # Return the top-k results.
        return [
            {
                **self.chunks[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices[:top_k]
        ]