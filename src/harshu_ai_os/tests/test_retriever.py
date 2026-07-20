import pytest
from harshu_ai_os.rag.retriever import retrieve_best_note


class FakeEmbedding:
    def __init__(self, values):
        self.values = values

def test_retrieve_best_note_rejects_empty_notes():
    client = FakeClient()

    with pytest.raises(
        ValueError,
        match="At least one note is required.",
    ):
        retrieve_best_note(
            client,
            "Which note is relevant?",
            [],
        )
class FakeResponse:
    def __init__(self, values):
        self.embeddings = [FakeEmbedding(values)]


class FakeModels:
    def embed_content(self, model, contents):
        vectors = {
            "Which note is about routing?": [1.0, 0.0],
            "FastAPI exposes the endpoint.": [0.0, 1.0],
            "The router selects a model.": [1.0, 0.0],
        }

        return FakeResponse(vectors[contents])


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_retrieve_best_note_returns_router_note():
    client = FakeClient()

    notes = [
        "FastAPI exposes the endpoint.",
        "The router selects a model.",
    ]

    result = retrieve_best_note(
        client,
        "Which note is about routing?",
        notes,
    )

    assert result["text"] == "The router selects a model."
    assert result["index"] == 1
    assert result["score"] == 1.0