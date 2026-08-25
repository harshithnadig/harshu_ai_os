"""Print retrieval quality, abstention results, and enforce CI quality gates for synthetic cases."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from harshu_ai_os.evaluations.retrieval_cases import evaluation_cases
from harshu_ai_os.evaluations.retrieval_evaluator import (
    evaluate_abstention_thresholds,
    run_retrieval_evaluation,
)
from harshu_ai_os.evaluations.retrieval_metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    extract_ranks,
)
from harshu_ai_os.rag.chroma_store import DEFAULT_TOP_K, get_notes_collection
from harshu_ai_os.rag.embedding_client import EMBEDDING_MODEL, get_embedding_client
from harshu_ai_os.rag.ingestion import sync_knowledge_base


class _FakeEmbedding:
    def __init__(self, values: list[float]):
        self.values = values


class _FakeResponse:
    def __init__(self, values: list[float]):
        self.embeddings = [_FakeEmbedding(values)]


class _DeterministicEmbeddingModels:
    def __init__(self, dim: int = 3072):
        self.dim = dim

    def embed_content(self, model: str, contents: str) -> _FakeResponse:
        import numpy as np
        import re

        words = re.findall(r"\w+", contents.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return _FakeResponse(vec.tolist())


class DeterministicOfflineClient:
    """Self-contained deterministic embedding client for offline testing and CI gates."""

    def __init__(self, dim: int = 3072):
        self.models = _DeterministicEmbeddingModels(dim=dim)


def record_baseline_configuration(collection) -> dict:
    """Capture reproducible state."""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    all_docs = collection.get()
    indexed_chunk_count = len(all_docs.get("ids", []))
    ids_sorted = sorted(all_docs.get("ids", []))
    corpus_fingerprint = hashlib.sha256("".join(ids_sorted).encode("utf-8")).hexdigest()

    answerable = sum(1 for c in evaluation_cases if c["answerable"])

    return {
        "git_commit_sha": git_sha,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": 50,
        "retrieval_top_k": DEFAULT_TOP_K,
        "evaluation_case_count": len(evaluation_cases),
        "supported_case_count": answerable,
        "unsupported_case_count": len(evaluation_cases) - answerable,
        "indexed_chunk_count": indexed_chunk_count,
        "corpus_fingerprint": corpus_fingerprint,
        "python_version": sys.version.split(" ")[0],
    }


def print_metrics(results_data: dict, run_name: str) -> dict:
    results = results_data["results"]
    answerable_results = [r for r in results if r["answerable"]]
    ranks = extract_ranks(answerable_results)

    hit_1 = calculate_hit_rate(ranks, k=1)
    hit_3 = calculate_hit_rate(ranks, k=3)
    hit_5 = calculate_hit_rate(ranks, k=5)
    mrr = calculate_mrr(ranks)

    p5_list = []
    r5_list = []
    for r in results:
        if r["answerable"] and r.get("expected_chunk_ids"):
            p = calculate_precision_at_k(r["retrieved_ids"], r["expected_chunk_ids"], k=5)
            rec = calculate_recall_at_k(r["retrieved_ids"], r["expected_chunk_ids"], k=5)
            p5_list.append(p)
            r5_list.append(rec)

    p5 = sum(p5_list) / len(p5_list) if p5_list else 0.0
    r5 = sum(r5_list) / len(r5_list) if r5_list else 0.0

    latencies = [r["latency"] for r in results if r.get("latency")]
    latencies.sort()

    if latencies:
        p50 = latencies[int((len(latencies) - 1) * 0.50)]
        p95 = latencies[int((len(latencies) - 1) * 0.95)]
    else:
        p50 = 0.0
        p95 = 0.0

    print(f"\n=== Metrics for {run_name} ===")
    print(f"Hit@1: {hit_1:.2f}%")
    print(f"Hit@3: {hit_3:.2f}%")
    print(f"Hit@5: {hit_5:.2f}%")
    print(f"MRR: {mrr:.4f}")
    print(f"Precision@5: {p5:.4f}")
    print(f"Recall@5: {r5:.4f}")
    print(f"Latency p50: {p50:.4f}s")
    print(f"Latency p95: {p95:.4f}s")

    category_metrics = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in category_metrics:
            category_metrics[cat] = {"ranks": [], "p5": [], "r5": [], "has_pk": False}

        if r["answerable"]:
            category_metrics[cat]["ranks"].append(r["rank"])
            if r.get("expected_chunk_ids"):
                p = calculate_precision_at_k(r["retrieved_ids"], r["expected_chunk_ids"], k=5)
                rec = calculate_recall_at_k(r["retrieved_ids"], r["expected_chunk_ids"], k=5)
                category_metrics[cat]["p5"].append(p)
                category_metrics[cat]["r5"].append(rec)
                category_metrics[cat]["has_pk"] = True

    print("\n--- BY CATEGORY ---")
    for cat, data in category_metrics.items():
        if data["ranks"]:
            cat_hit_3 = f"{calculate_hit_rate(data['ranks'], k=3):.2f}%"
        else:
            cat_hit_3 = "N/A"

        if data["has_pk"]:
            cat_p5 = f"{sum(data['p5']) / len(data['p5']):.4f}"
            cat_r5 = f"{sum(data['r5']) / len(data['r5']):.4f}"
        else:
            cat_p5 = "N/A"
            cat_r5 = "N/A"

        print(f"Category: {cat:<30} Hit@3: {cat_hit_3:<7} P@5: {cat_p5:<7} R@5: {cat_r5:<7}")

    accuracy = results_data.get("summary", {}).get("retrieval_accuracy", 0.0)

    return {
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_5": hit_5,
        "mrr": mrr,
        "p5": p5,
        "r5": r5,
        "accuracy": accuracy,
    }


def get_evaluation_resources(force_offline: bool = False):
    """Obtain or initialize the vector collection and embedding client."""
    if not force_offline:
        try:
            client = get_embedding_client()
            # Test connectivity
            client.post("/embeddings", json={"model": EMBEDDING_MODEL, "input": "ping"}, timeout=2.0)
            collection = get_notes_collection()
            if collection.count() == 0:
                docs_path = Path("examples/documents")
                if docs_path.exists():
                    sync_knowledge_base(collection, client, docs_path, chunk_size=50)
            return collection, client, "OmniRoute Gateway"
        except Exception:
            pass

    # Self-contained offline mode
    import chromadb

    client = DeterministicOfflineClient()
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="harshu_ai_os_ci_eval",
        metadata={"hnsw:space": "cosine"},
    )
    docs_path = Path("examples/documents")
    if docs_path.exists():
        sync_knowledge_base(collection, client, docs_path, chunk_size=50)

    return collection, client, "Deterministic Offline Client"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG retrieval quality evaluation and CI quality gate.")
    parser.add_argument(
        "--min-hit3",
        type=float,
        default=float(os.getenv("RAG_MIN_HIT3", "25.0")),
        help="Minimum required Hit@3 percentage (default: 25.0)",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=float(os.getenv("RAG_MIN_MRR", "0.15")),
        help="Minimum required Mean Reciprocal Rank (default: 0.15)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=float(os.getenv("RAG_MIN_ACCURACY", "30.0")),
        help="Minimum required retrieval accuracy percentage (default: 30.0)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force deterministic offline evaluation without external network requests",
    )
    args = parser.parse_args()

    collection, client, client_name = get_evaluation_resources(force_offline=args.offline)

    print(f"Using Embedding Provider: {client_name}")
    config = record_baseline_configuration(collection)
    print("=== CONFIGURATION ===")
    for k, v in config.items():
        print(f"{k}: {v}")

    print("\nRunning Evaluation #1...")
    results_1 = run_retrieval_evaluation(collection, client, evaluation_cases)
    metrics_1 = print_metrics(results_1, "Run 1")

    print("\nRunning Evaluation #2...")
    results_2 = run_retrieval_evaluation(collection, client, evaluation_cases)
    metrics_2 = print_metrics(results_2, "Run 2")

    print("\n=== REPRODUCIBILITY CHECK ===")
    matches = True
    for key in ("hit_1", "hit_3", "hit_5", "mrr", "p5", "r5"):
        if abs(metrics_1[key] - metrics_2[key]) > 1e-6:
            matches = False
            print(f"MISMATCH in {key}: {metrics_1[key]} vs {metrics_2[key]}")
    if matches:
        print("All quality metrics exactly match between runs.")
    else:
        print("Quality metrics do not perfectly match.")

    sweep_results = evaluate_abstention_thresholds(collection, client, evaluation_cases)
    print("\n--- ABSTENTION THRESHOLD SWEEP ---")
    print(
        f"{'Threshold':<10} {'Correct Gen':<14} {'Correct Abs':<14} "
        f"{'False Accept':<14} {'False Abs':<14}"
    )
    for result in sweep_results:
        print(
            f"{result['threshold']:<10.2f} "
            f"{result['correct_generations']:<14} "
            f"{result['correct_abstentions']:<14} "
            f"{result['false_accepts']:<14} "
            f"{result['false_abstentions']:<14}"
        )

    print("\n=== QUALITY GATE VERIFICATION ===")
    failures = []
    if not matches:
        failures.append("Reproducibility check failed between Run 1 and Run 2.")
    if metrics_1["hit_3"] < args.min_hit3:
        failures.append(f"Hit@3 ({metrics_1['hit_3']:.2f}%) is below minimum threshold ({args.min_hit3:.2f}%).")
    if metrics_1["mrr"] < args.min_mrr:
        failures.append(f"MRR ({metrics_1['mrr']:.4f}) is below minimum threshold ({args.min_mrr:.4f}).")
    if metrics_1["accuracy"] < args.min_accuracy:
        failures.append(f"Retrieval accuracy ({metrics_1['accuracy']:.2f}%) is below minimum threshold ({args.min_accuracy:.2f}%).")

    if failures:
        print("\n[AI QUALITY GATE FAILED]")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print("\n[AI QUALITY GATE PASSED]")
        print(f"  - Hit@3: {metrics_1['hit_3']:.2f}% (threshold: >= {args.min_hit3:.2f}%)")
        print(f"  - MRR: {metrics_1['mrr']:.4f} (threshold: >= {args.min_mrr:.4f})")
        print(f"  - Accuracy: {metrics_1['accuracy']:.2f}% (threshold: >= {args.min_accuracy:.2f}%)")
        print("  - Reproducibility: Passed (100% deterministic)")
        sys.exit(0)


if __name__ == "__main__":
    main()
