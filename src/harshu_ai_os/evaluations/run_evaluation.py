"""Print retrieval quality and abstention results for the synthetic cases."""

import hashlib
import sys
import subprocess

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
from harshu_ai_os.rag.chroma_store import get_notes_collection, DEFAULT_TOP_K
from harshu_ai_os.rag.embedding_client import get_embedding_client, EMBEDDING_MODEL

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
        
    return {
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_5": hit_5,
        "mrr": mrr,
        "p5": p5,
        "r5": r5
    }

def main() -> None:
    client = get_embedding_client()
    collection = get_notes_collection()
    
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
    for key in metrics_1:
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

if __name__ == "__main__":
    main()
