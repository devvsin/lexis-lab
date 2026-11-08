def recall_at_k(
    results: list[dict],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """Measure whether any relevant chunk appears in the top-k results."""
    if not relevant_chunk_ids:
        return 0.0

    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    return float(
        bool(retrieved_ids.intersection(relevant_chunk_ids))
    )


def retrieval_hit(
    results: list[dict],
    relevant_chunk_ids: list[str],
    k: int,
) -> bool:
    """Return whether the top-k results contain a relevant chunk."""
    if not relevant_chunk_ids:
        return False

    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    return bool(
        retrieved_ids.intersection(relevant_chunk_ids)
    )