from harshu_ai_os.rag.service import answer_with_rag
import pytest

def test_answer_with_rag_rejects_empty_question_before_dependencies():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        answer_with_rag(
            object(),
            "   ",
            ["This note must never be embedded."],
            lambda prompt: "This must never run.",
        )

class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResponse:
    def __init__(self, values):
        self.embeddings = [FakeEmbedding(values)]


class FakeModels:
    def embed_content(self, model, contents):
        vectors = {
            "How is Harshu AI OS tested?": [1.0, 0.0],
            "FastAPI exposes the endpoint.": [0.0, 1.0],
            "Harshu AI OS is tested using Pytest.": [1.0, 0.0],
        }

        return FakeResponse(vectors[contents])


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_answer_with_rag_returns_answer_and_retrieval_evidence():
    client = FakeClient()

    notes = [
        "FastAPI exposes the endpoint.",
        "Harshu AI OS is tested using Pytest.",
    ]

    def fake_generate_text(prompt):
        assert "Harshu AI OS is tested using Pytest." in prompt
        return "Harshu AI OS is tested using Pytest."

    result = answer_with_rag(
        client,
        "How is Harshu AI OS tested?",
        notes,
        fake_generate_text,
    )

    assert result["answer"] == "Harshu AI OS is tested using Pytest."
    assert result["context"] == "Harshu AI OS is tested using Pytest."
    assert result["index"] == 1
    assert result["score"] == 1.0