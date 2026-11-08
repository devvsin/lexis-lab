
from evaluation.runner import evaluate_retriever


class FakeRetriever:
    def search(self, query, top_k=3):
        if query == "answerable":
            return [{"chunk_id": "chunk-1"}]

        return []

    
def test_evaluate_retriever() -> None:
    dataset = [
        {
            "question": "answerable",
            "relevant_chunk_ids": ["chunk-1"],
            "answerable": True,
        },
        {
            "question": "unanswerable",
            "relevant_chunk_ids": [],
            "answerable": False,
        },
    ]

    result = evaluate_retriever(
        FakeRetriever(),
        dataset=dataset,
        k=1,
    )

    assert result["recall_at_k"] == 1.0
    assert result["answerable_questions"] == 1
    assert result["unanswerable_questions"] == 1
    assert result["unanswerable_retrieval_rate"] == 0.0
