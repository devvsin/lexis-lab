from evaluation.retrieval import recall_at_k


def test_recall_at_k_when_relevant_chunk_is_retrieved() -> None:
    results = [
        {"chunk_id": "chunk-1"},
        {"chunk_id": "chunk-2"},
    ]

    assert recall_at_k(
        results,
        ["chunk-1"],
        k=1,
    ) == 1.0


def test_recall_at_k_when_relevant_chunk_is_missing() -> None:
    results = [
        {"chunk_id": "chunk-2"},
    ]

    assert recall_at_k(
        results,
        ["chunk-1"],
        k=1,
    ) == 0.0