"""Unit tests for optional CrossEncoder reranking."""

from unittest.mock import MagicMock, patch

import pytest

from harshu_ai_os.rag.reranker import rerank_candidates


def candidate_data() -> dict[str, list]:
    """Return small synthetic candidates shared by the tests."""
    return {
        "ids": ["id_1", "id_2", "id_3"],
        "texts": ["Text 1 (low)", "Text 2 (high)", "Text 3 (mid)"],
        "distances": [0.11, 0.22, 0.33],
        "metadatas": [
            {"source": "doc1"},
            {"source": "doc2"},
            {"source": "doc3"},
        ],
    }


def test_empty_candidates_return_empty_result():
    result = rerank_candidates(
        "What is FastAPI?",
        {"ids": [], "texts": [], "distances": [], "metadatas": []},
    )

    assert result == {
        "ids": [],
        "texts": [],
        "distances": [],
        "metadatas": [],
        "reranker_scores": [],
    }


@patch("harshu_ai_os.rag.reranker.get_reranker_model")
def test_reranking_keeps_each_chunks_fields_together(mock_get_model):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.9, 0.5]
    mock_get_model.return_value = mock_model

    result = rerank_candidates("Query", candidate_data(), top_k=2)

    assert result["ids"] == ["id_2", "id_3"]
    assert result["texts"] == ["Text 2 (high)", "Text 3 (mid)"]
    assert result["distances"] == [0.22, 0.33]
    assert result["metadatas"] == [{"source": "doc2"}, {"source": "doc3"}]
    assert result["reranker_scores"] == [0.9, 0.5]


def test_empty_question_returns_empty_result():
    result = rerank_candidates("   ", candidate_data())
    assert result["ids"] == []


def test_non_positive_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        rerank_candidates("Query", candidate_data(), top_k=0)


def test_misaligned_candidate_fields_are_rejected():
    candidates = candidate_data()
    candidates["ids"] = ["only-one-id"]

    with pytest.raises(ValueError, match="same number"):
        rerank_candidates("Query", candidates)
