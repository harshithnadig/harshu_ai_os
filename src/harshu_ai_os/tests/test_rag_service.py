import pytest
from harshu_ai_os.rag.service import (
    answer_with_chroma_rag,
    generate_grounded_answer,
    should_abstain,
)
from harshu_ai_os.rag.sufficiency_judge import SufficiencyVerdict
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from harshu_ai_os.llm.exceptions import LLMServiceError


from harshu_ai_os.rag.chroma_store import DEFAULT_TOP_K


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
            "Does ChromaDB automatically handle user password hashing?": [1.0, 0.0],
        }

        return FakeResponse(vectors[contents])


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


class FakeCollection:
    def query(self, query_embeddings, n_results):
        assert query_embeddings == [[1.0, 0.0]]
        assert n_results == DEFAULT_TOP_K

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


class HighDistanceCollection:
    def query(self, query_embeddings, n_results):
        return {
            "ids": [["note-1", "note-0"]],
            "documents": [
                [
                    "Harshu AI OS is tested using Pytest.",
                    "FastAPI exposes the endpoint.",
                ]
            ],
            "distances": [[0.8, 1.2]],
            "metadatas": [
                [
                    {"source": "manual", "position": 1},
                    {"source": "manual", "position": 0},
                ]
            ],
        }


class ExactDistanceCollection:
    def query(self, query_embeddings, n_results):
        return {
            "ids": [["note-1", "note-0"]],
            "documents": [
                [
                    "Harshu AI OS is tested using Pytest.",
                    "FastAPI exposes the endpoint.",
                ]
            ],
            "distances": [[0.5, 1.0]],
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

    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=True,
            reason="Supported",
            supporting_chunk_ids=["note-1", "note-0"],
        )

    def fake_model_factory(received_route):
        assert received_route == route
        return FakeListChatModel(responses=["Harshu AI OS is tested using Pytest."])

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.create_chat_model_from_route",
        fake_model_factory,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "Harshu AI OS is tested using Pytest."
    assert result["abstained"] is False
    assert result["abstention_reason"] is None
    assert result["judge_reason"] == "Supported"
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


def test_answer_with_chroma_rag_filters_to_supporting_chunks_only(monkeypatch):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    # note-1 is supporting, note-0 is not
    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=True,
            reason="Only note-1 supports the question",
            supporting_chunk_ids=["note-1"],
        )

    def fake_generate(route, question, context):
        assert context == "Harshu AI OS is tested using Pytest."
        return "Tested using Pytest."

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        fake_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["abstained"] is False
    assert result["context"] == "Harshu AI OS is tested using Pytest."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "note-1"


def test_answer_with_chroma_rag_abstains_on_near_match_password_hashing(monkeypatch):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=False,
            reason="Context mentions ChromaDB but does not state password hashing.",
            supporting_chunk_ids=[],
        )

    def failing_generate(*args, **kwargs):
        raise AssertionError("LLM generation must not be called when judge rejects")

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        failing_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "Does ChromaDB automatically handle user password hashing?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "I do not have enough information."
    assert result["abstained"] is True
    assert result["abstention_reason"] == "insufficient_context"
    assert result["judge_reason"] == "Context mentions ChromaDB but does not state password hashing."
    assert result["citations"] == []


def test_answer_with_chroma_rag_abstains_on_invalid_verdict_empty_supporting_ids(monkeypatch):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    # Invalid: answerable=True but supporting_chunk_ids is empty
    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=True,
            reason="Claims answerable but gave no IDs",
            supporting_chunk_ids=[],
        )

    def failing_generate(*args, **kwargs):
        raise AssertionError("LLM generation must not be called on invalid verdict")

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        failing_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "I do not have enough information."
    assert result["abstained"] is True
    assert result["abstention_reason"] == "insufficient_context"
    assert result["citations"] == []


def test_answer_with_chroma_rag_abstains_on_invalid_verdict_unknown_chunk_id(monkeypatch):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    # Invalid: answerable=True but ID does not exist in retrieval["ids"]
    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=True,
            reason="Invented chunk ID",
            supporting_chunk_ids=["invented-chunk-99"],
        )

    def failing_generate(*args, **kwargs):
        raise AssertionError("LLM generation must not be called when supporting IDs are invalid")

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        failing_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "I do not have enough information."
    assert result["abstained"] is True
    assert result["abstention_reason"] == "insufficient_context"
    assert result["citations"] == []


def test_answer_with_chroma_rag_abstains_on_invalid_verdict_unanswerable_with_ids(monkeypatch):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    # Invalid: answerable=False but supporting_chunk_ids is non-empty
    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=False,
            reason="Contradictory verdict",
            supporting_chunk_ids=["note-1"],
        )

    def failing_generate(*args, **kwargs):
        raise AssertionError("LLM generation must not be called on contradictory verdict")

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        failing_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "I do not have enough information."
    assert result["abstained"] is True
    assert result["abstention_reason"] == "insufficient_context"
    assert result["citations"] == []


def test_answer_with_chroma_rag_propagates_judge_provider_errors(monkeypatch):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    def failing_judge(route, question, chunks, chunk_ids):
        raise LLMServiceError("AI service is temporarily unavailable. Please try again.")

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        failing_judge,
    )

    with pytest.raises(LLMServiceError, match="AI service is temporarily unavailable"):
        answer_with_chroma_rag(
            collection,
            client,
            "How is Harshu AI OS tested?",
            route,
            maximum_distance=0.5,
        )


def test_answer_with_chroma_rag_abstains_when_best_distance_above_threshold(
    monkeypatch,
):
    collection = HighDistanceCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    def failing_generate(*args, **kwargs):
        raise AssertionError("LLM generation must not be called when abstaining")

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        failing_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "I do not have enough information."
    assert result["abstained"] is True
    assert result["abstention_reason"] == "insufficient_context"
    assert result["citations"] == []
    assert result["distances"] == [0.8, 1.2]
    assert result["context"] == (
        "Harshu AI OS is tested using Pytest.\n\nFastAPI exposes the endpoint."
    )


def test_answer_with_chroma_rag_abstains_when_no_distances(
    monkeypatch,
):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    def failing_generate(*args, **kwargs):
        raise AssertionError("LLM generation must not be called when abstaining")

    def fake_empty_query_notes(coll, cli, quest):
        return {"ids": [], "texts": [], "distances": [], "metadatas": []}

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.query_notes",
        fake_empty_query_notes,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        failing_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "I do not have enough information."
    assert result["abstained"] is True
    assert result["abstention_reason"] == "insufficient_context"
    assert result["citations"] == []
    assert result["distances"] == []


def test_answer_with_chroma_rag_generates_when_best_distance_equals_threshold(
    monkeypatch,
):
    collection = ExactDistanceCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=True,
            reason="Supported",
            supporting_chunk_ids=["note-1", "note-0"],
        )

    def fake_model_factory(received_route):
        return FakeListChatModel(responses=["Harshu AI OS is tested using Pytest."])

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.create_chat_model_from_route",
        fake_model_factory,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "Harshu AI OS is tested using Pytest."
    assert result["abstained"] is False
    assert result["abstention_reason"] is None
    assert len(result["citations"]) == 2


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


class MultiCandidateCollection:
    def query(self, query_embeddings, n_results):
        assert n_results == 4
        return {
            "ids": [["chunk-1", "chunk-2", "chunk-3", "chunk-4"]],
            "documents": [
                [
                    "Text 1 (low rerank score)",
                    "Text 2 (top rerank score)",
                    "Text 3 (lowest rerank score)",
                    "Text 4 (second rerank score)",
                ]
            ],
            "distances": [[0.1, 0.4, 0.2, 0.3]],
            "metadatas": [
                [
                    {"source": "doc1", "chunk_index": 0},
                    {"source": "doc2", "chunk_index": 1},
                    {"source": "doc3", "chunk_index": 2},
                    {"source": "doc4", "chunk_index": 3},
                ]
            ],
        }


def test_answer_with_chroma_rag_with_reranking_reorders_and_limits_chunks(
    monkeypatch,
):
    collection = MultiCandidateCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    # Mock reranker to return chunks reordered by relevance score (top_k=2 -> chunk-2, chunk-4)
    def fake_rerank(question, candidates, top_k=5, model_name=None):
        assert question == "How is Harshu AI OS tested?"
        assert top_k == 2
        assert len(candidates["ids"]) == 4
        return {
            "ids": ["chunk-2", "chunk-4"],
            "texts": [
                "Text 2 (top rerank score)",
                "Text 4 (second rerank score)",
            ],
            "distances": [0.4, 0.3],
            "metadatas": [
                {"source": "doc2", "chunk_index": 1},
                {"source": "doc4", "chunk_index": 3},
            ],
            "reranker_scores": [0.95, 0.85],
        }

    def fake_judge(route, question, chunks, chunk_ids):
        assert chunk_ids == ["chunk-2", "chunk-4"]
        assert chunks == [
            "Text 2 (top rerank score)",
            "Text 4 (second rerank score)",
        ]
        return SufficiencyVerdict(
            answerable=True,
            reason="Chunk 2 directly answers.",
            supporting_chunk_ids=["chunk-2"],
        )

    def fake_generate(route, question, context):
        assert context == "Text 2 (top rerank score)"
        return "Answer based on chunk 2."

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.rerank_candidates",
        fake_rerank,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        fake_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
        enable_reranking=True,
        candidate_top_k=4,
        top_k=2,
    )

    assert result["answer"] == "Answer based on chunk 2."
    assert result["abstained"] is False
    assert result["ids"] == ["chunk-2", "chunk-4"]
    assert result["distances"] == [0.4, 0.3]
    assert result["reranker_scores"] == [0.95, 0.85]
    assert result["context"] == "Text 2 (top rerank score)"
    # Citation must only contain the supporting chunk (chunk-2)
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk-2"
    assert result["citations"][0]["distance"] == 0.4
    assert result["citations"][0]["source"] == "doc2"


def test_answer_with_chroma_rag_reranking_disabled_by_default(monkeypatch):
    collection = FakeCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    def failing_rerank(*args, **kwargs):
        raise AssertionError("Reranker must not be called when enable_reranking is False")

    def fake_judge(route, question, chunks, chunk_ids):
        return SufficiencyVerdict(
            answerable=True,
            reason="Supported",
            supporting_chunk_ids=["note-1"],
        )

    def fake_generate(route, question, context):
        return "Harshu AI OS is tested using Pytest."

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.rerank_candidates",
        failing_rerank,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        fake_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
    )

    assert result["answer"] == "Harshu AI OS is tested using Pytest."
    assert result["ids"] == ["note-1", "note-0"]
    assert "reranker_scores" not in result


def test_answer_with_chroma_rag_reranking_fallback_on_failure(monkeypatch):
    collection = MultiCandidateCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    def failing_rerank(*args, **kwargs):
        raise RuntimeError("Reranking is optional. Install it with: uv sync --extra reranking")

    def fake_judge(route, question, chunks, chunk_ids):
        # Fallback should pass top_k=2 Chroma chunks (chunk-1, chunk-2)
        assert chunk_ids == ["chunk-1", "chunk-2"]
        return SufficiencyVerdict(
            answerable=True,
            reason="Supported by fallback top chunks.",
            supporting_chunk_ids=["chunk-1"],
        )

    def fake_generate(route, question, context):
        return "Fallback answer."

    monkeypatch.setattr(
        "harshu_ai_os.rag.service.rerank_candidates",
        failing_rerank,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        fake_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
        enable_reranking=True,
        candidate_top_k=4,
        top_k=2,
    )

    assert result["answer"] == "Fallback answer."
    assert result["abstained"] is False
    assert result["ids"] == ["chunk-1", "chunk-2"]
    assert result["distances"] == [0.1, 0.4]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk-1"


def test_answer_with_chroma_rag_end_to_end_with_mocked_cross_encoder(monkeypatch):
    """Test full integration path using real rerank_candidates with mocked model."""
    from unittest.mock import MagicMock

    collection = MultiCandidateCollection()
    client = FakeClient()
    route = {"model": "fake/model", "max_tokens": 100}

    mock_model = MagicMock()
    # 4 candidates: chunk-1 (0.2), chunk-2 (0.95), chunk-3 (0.1), chunk-4 (0.8)
    mock_model.predict.return_value = [0.2, 0.95, 0.1, 0.8]

    def fake_judge(route, question, chunks, chunk_ids):
        assert chunk_ids == ["chunk-2", "chunk-4"]
        return SufficiencyVerdict(
            answerable=True,
            reason="Supported by chunk-2",
            supporting_chunk_ids=["chunk-2"],
        )

    def fake_generate(route, question, context):
        assert context == "Text 2 (top rerank score)"
        return "Grounded answer from reranked chunk 2."

    monkeypatch.setattr(
        "harshu_ai_os.rag.reranker.get_reranker_model",
        lambda *args, **kwargs: mock_model,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.judge_context_sufficiency",
        fake_judge,
    )
    monkeypatch.setattr(
        "harshu_ai_os.rag.service.generate_grounded_answer",
        fake_generate,
    )

    result = answer_with_chroma_rag(
        collection,
        client,
        "How is Harshu AI OS tested?",
        route,
        maximum_distance=0.5,
        enable_reranking=True,
        candidate_top_k=4,
        top_k=2,
    )

    assert result["answer"] == "Grounded answer from reranked chunk 2."
    assert result["abstained"] is False
    assert result["ids"] == ["chunk-2", "chunk-4"]
    assert result["distances"] == [0.4, 0.3]
    assert result["reranker_scores"] == [0.95, 0.8]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk-2"
    assert result["citations"][0]["distance"] == 0.4
    assert result["citations"][0]["source"] == "doc2"


