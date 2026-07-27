import pytest
from harshu_ai_os.rag.service import (
    answer_with_chroma_rag,
    generate_grounded_answer,
    should_abstain,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from harshu_ai_os.llm.exceptions import LLMServiceError


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


class FakeCollection:
    def query(self, query_embeddings, n_results):
        assert query_embeddings == [[1.0, 0.0]]
        assert n_results == 3

        return {
            "ids": [["note-1", "note-0"]],
            "documents": [
                [
                    "Harshu AI OS is tested using Pytest.",
                    "FastAPI exposes the endpoint.",
                ]
            ],
            "distances": [[0.0, 1.0]],
            "metadatas": [
                [
                    {"source": "manual", "position": 1},
                    {"source": "manual", "position": 0},
                ]
            ],
        }


def test_answer_with_chroma_rag_returns_grounded_answer_and_evidence(
    monkeypatch,
):
    collection = FakeCollection()
    client = FakeClient()
    route = {
        "model": "fake/model",
        "max_tokens": 100,
    }

    def fake_model_factory(received_route):
        assert received_route == route
        return FakeListChatModel(responses=["Harshu AI OS is tested using Pytest."])

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.create_chat_model_from_route",
        fake_model_factory,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
    )

    assert result["answer"] == "Harshu AI OS is tested using Pytest."
    assert result["ids"] == ["note-1", "note-0"]
    assert result["distances"] == [0.0, 1.0]
    assert result["metadatas"][0]["position"] == 1
    assert result["context"] == (
        "Harshu AI OS is tested using Pytest.\n\nFastAPI exposes the endpoint."
    )
    assert result["citations"] == [
        {
            "source": "manual",
            "chunk_id": "note-1",
            "chunk_index": None,
            "distance": 0.0,
        },
        {
            "source": "manual",
            "chunk_id": "note-0",
            "chunk_index": None,
            "distance": 1.0,
        },
    ]


def test_should_abstain_uses_the_best_distance() -> None:
    assert should_abstain([], maximum_distance=0.5) is True
    assert should_abstain([0.7, 0.2], maximum_distance=0.5) is False


def test_grounded_generation_translates_provider_errors(monkeypatch) -> None:
    def failing_model_factory(route):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.create_chat_model_from_route",
        failing_model_factory,
    )

    with pytest.raises(
        LLMServiceError,
        match="AI service is temporarily unavailable",
    ):
        generate_grounded_answer(
            {
                "model": "fake/model",
                "max_tokens": 100,
            },
            "What does ChromaDB do?",
            "ChromaDB stores embeddings.",
        )
