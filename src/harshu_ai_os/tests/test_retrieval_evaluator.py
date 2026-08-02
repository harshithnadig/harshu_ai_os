from harshu_ai_os.evaluations.retrieval_evaluator import evaluate_retrieval


def test_evaluate_retrieval_returns_true_when_expected_text_exists():
    retrieved_chunks = [
        "ChromaDB stores document embeddings and retrieves relevant notes."
    ]

    result = evaluate_retrieval(
        "Chroma",
        retrieved_chunks,
    )

    assert result is True



def test_evaluate_retrieval_returns_false_when_expected_text_missing():
    retrieved_chunks = [
        "FastAPI exposes application endpoints."
    ]

    result = evaluate_retrieval(
        "Chroma",
        retrieved_chunks,
    )

    assert result is False