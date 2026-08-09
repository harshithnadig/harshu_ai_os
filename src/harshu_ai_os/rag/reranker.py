"""Optional second-stage reranking for retrieved RAG chunks.

Learning flow:
    Chroma candidates -> question/chunk pairs -> relevance scores -> best chunks

This module is an experiment. The live ``/ask/rag`` endpoint still uses the
simpler Chroma retrieval path in ``rag/service.py``.
"""

from functools import lru_cache
from typing import Any

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RESULT_FIELDS = ("ids", "texts", "distances", "metadatas")


def empty_result() -> dict[str, list]:
    """Return the same predictable shape used by successful reranking."""
    return {**{field: [] for field in RESULT_FIELDS}, "reranker_scores": []}


@lru_cache(maxsize=2)
def get_reranker_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load a CrossEncoder once, then reuse it for later questions."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "Reranking is optional. Install it with: uv sync --extra reranking"
        ) from error

    return CrossEncoder(model_name)


def rerank_candidates(
    question: str,
    candidates: dict[str, list],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, list]:
    """Resort retrieved chunks by how well each chunk answers the question.

    Chroma quickly finds possible matches. A CrossEncoder then reads each
    question/chunk pair more carefully and gives it a relevance score.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    columns = {field: candidates.get(field, []) for field in RESULT_FIELDS}
    if not columns["texts"] or not question.strip():
        return empty_result()

    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Candidate fields must contain the same number of items.")

    pairs = [(question, text) for text in columns["texts"]]
    raw_scores = get_reranker_model(model_name).predict(pairs)
    scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

    if len(scores) != len(columns["texts"]):
        raise ValueError("The reranker must return one score for every candidate.")

    # zip keeps every score attached to the chunk it describes while sorting.
    scored_items = [
        {
            "id": chunk_id,
            "text": text,
            "distance": distance,
            "metadata": metadata,
            "score": float(score),
        }
        for chunk_id, text, distance, metadata, score in zip(
            columns["ids"],
            columns["texts"],
            columns["distances"],
            columns["metadatas"],
            scores,
            strict=True,
        )
    ]
    scored_items.sort(key=lambda item: item["score"], reverse=True)
    best_items = scored_items[:top_k]

    return {
        "ids": [item["id"] for item in best_items],
        "texts": [item["text"] for item in best_items],
        "distances": [item["distance"] for item in best_items],
        "metadatas": [item["metadata"] for item in best_items],
        "reranker_scores": [item["score"] for item in best_items],
    }
