"""Print retrieval quality and abstention results for the synthetic cases."""

from harshu_ai_os.evaluations.retrieval_cases import evaluation_cases
from harshu_ai_os.evaluations.retrieval_evaluator import (
    evaluate_abstention_thresholds,
    run_retrieval_evaluation,
)
from harshu_ai_os.evaluations.retrieval_metrics import (
    calculate_hit_rate,
    calculate_mrr,
    extract_ranks,
)
from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client


def main() -> None:
    """Run the complete local retrieval evaluation report."""
    client = get_embedding_client()
    collection = get_notes_collection()
    results = run_retrieval_evaluation(collection, client, evaluation_cases)

    print(results)

    summary = results["summary"]
    print(
        f"\nRetrieval Accuracy: {summary['retrieval_accuracy']:.2f}% "
        f"({summary['retrieval_passed']}/{summary['answerable_cases']} passed)"
    )

    # Ranks preserve misses as None, because a missed query must earn zero.
    ranks = extract_ranks(results["results"])
    print(f"Ranks: {ranks}")
    print(f"Hit@3 rate: {calculate_hit_rate(ranks, k=3):.2f}%")
    print(f"MRR: {calculate_mrr(ranks):.4f}")

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
