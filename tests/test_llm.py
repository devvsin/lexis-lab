from generation.llm import generate_answer


def test_llm_refuses_without_evidence() -> None:
    result = generate_answer(
        "What is the termination period?",
        [],
    )

    assert result.insufficient_evidence is True
    assert result.confidence == 0.0
    assert result.sources == []