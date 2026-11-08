from retrieval.hybrid import HybridRetriever


def test_hybrid_retriever_returns_results() -> None:
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "contract-1",
            "page_number": 2,
            "text": "The agreement may be terminated with thirty days notice.",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "contract-1",
            "page_number": 5,
            "text": "Payment shall be made within fifteen business days.",
        },
    ]

    retriever = HybridRetriever(chunks)

    results = retriever.search(
        "How can the agreement be ended?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert "bm25_score" in results[0]
    assert "semantic_score" in results[0]