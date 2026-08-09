"""Compare normal Chroma retrieval with optional CrossEncoder reranking.

Run this experiment only after the basic rank metrics make sense. It uses the
existing synthetic evaluation cases and reports both quality and latency.
"""

from time import perf_counter

import numpy as np

from harshu_ai_os.evaluations.retrieval_cases import evaluation_cases
from harshu_ai_os.evaluations.retrieval_evaluator import find_expected_evidence
from harshu_ai_os.evaluations.retrieval_metrics import calculate_hit_rate, calculate_mrr
from harshu_ai_os.rag.chroma_store import get_notes_collection, query_notes
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.reranker import (
    DEFAULT_MODEL_NAME,
    get_reranker_model,
    rerank_candidates,
)


def print_quality(label: str, ranks: list[int | None]) -> None:
    """Print the retrieval metrics currently being learned."""
    print(f"\n--- {label} ---")
    for k in (1, 3, 5):
        print(f"Hit@{k}: {calculate_hit_rate(ranks, k=k):.2f}%")
    print(f"MRR:   {calculate_mrr(ranks):.4f}")


def print_latency(label: str, values: list[float]) -> None:
    """Print average and slow-tail latency in milliseconds."""
    print(f"Average {label}: {np.mean(values):.2f} ms")
    print(f"p95 {label}:     {np.percentile(values, 95):.2f} ms")


def run_experiment(model_name: str = DEFAULT_MODEL_NAME) -> None:
    """Measure whether slower reranking improves evidence positions."""
    client = get_embedding_client()
    collection = get_notes_collection()

    print(f"Loading optional reranker: {model_name}")
    started_at = perf_counter()
    model = get_reranker_model(model_name)
    model.predict([("warmup question", "warmup passage")])
    print(f"Model ready in {(perf_counter() - started_at) * 1000:.2f} ms")

    baseline_ranks: list[int | None] = []
    reranked_ranks: list[int | None] = []
    retrieval_times: list[float] = []
    reranking_times: list[float] = []

    print(f"Running {len(evaluation_cases)} synthetic cases...")
    for case in evaluation_cases:
        retrieval_started_at = perf_counter()
        candidates = query_notes(collection, client, case["question"], top_k=10)
        retrieval_times.append((perf_counter() - retrieval_started_at) * 1000)

        reranking_started_at = perf_counter()
        reranked = rerank_candidates(
            case["question"], candidates, top_k=5, model_name=model_name
        )
        reranking_times.append((perf_counter() - reranking_started_at) * 1000)

        # Unanswerable cases have no correct evidence rank to measure.
        if not case["answerable"]:
            continue

        expected = case["expected_evidence"]
        baseline_match = find_expected_evidence(
            expected, candidates["texts"][:5], candidates["metadatas"][:5]
        )
        reranked_match = find_expected_evidence(
            expected, reranked["texts"], reranked["metadatas"]
        )
        baseline_ranks.append(baseline_match["rank"])
        reranked_ranks.append(reranked_match["rank"])

    total_times = [
        retrieval_ms + reranking_ms
        for retrieval_ms, reranking_ms in zip(
            retrieval_times, reranking_times, strict=True
        )
    ]

    print("\nCROSS-ENCODER RERANKING EXPERIMENT")
    print(f"Model: {model_name}")
    print_quality("BASELINE: CHROMA TOP 5", baseline_ranks)
    print_quality("RERANKED: TOP 10 THEN BEST 5", reranked_ranks)
    print_latency("retrieval", retrieval_times)
    print_latency("reranking", reranking_times)
    print_latency("total", total_times)


if __name__ == "__main__":
    run_experiment()
