from retrieval.bm25 import BM25Retriever


def test_bm25_returns_relevant_chunk() -> None:
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "contract-1",
            "page_number": 2,
            "text": "The agreement may be terminated with 30 days notice.",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "contract-1",
            "page_number": 5,
            "text": "Payment shall be made within fifteen business days.",
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "contract-1",
            "page_number": 7,
            "text": "Confidential information must not be disclosed.",
        },
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("termination notice", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["page_number"] == 2
    assert results[0]["score"] > 0


def test_bm25_returns_empty_for_blank_query() -> None:
    retriever = BM25Retriever([])

    assert retriever.search("") == []


def test_bm25_returns_empty_for_unknown_terms() -> None:
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "contract-1",
            "page_number": 1,
            "text": "The agreement may be terminated.",
        }
    ]

    retriever = BM25Retriever(chunks)

    assert retriever.search("xyznonexistent", top_k=3) == []