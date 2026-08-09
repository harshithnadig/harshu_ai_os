"""Run retrieval cases and record where the expected evidence appeared."""

from harshu_ai_os.rag.chroma_store import query_notes
from harshu_ai_os.rag.service import should_abstain


def find_expected_evidence(expected, chunks, metadatas):
    """Return the first 1-based rank containing the expected text."""
    for rank, chunk in enumerate(chunks, start=1):
        if expected in chunk:
            return {
                "matched": True,
                "rank": rank,
                "source": metadatas[rank - 1]["source"],
                "chunk_text": chunk,
            }

    return {"matched": False, "rank": None, "source": None, "chunk_text": None}


def run_retrieval_evaluation(collection, client, evaluation_cases):
    """Retrieve every case and return both detailed results and a summary."""
    passed = 0
    case_results = []

    for case in evaluation_cases:
        case_id = case["id"]
        question = case["question"]
        answerable = case["answerable"]
        expected_evidence = case["expected_evidence"]

        if not answerable:
            case_results.append(
                {
                    "id": case_id,
                    "question": question,
                    "answerable": False,
                    "expected_evidence": None,
                    "evaluation": {
                        "status": "not_applicable",
                        "matched": None,
                        "rank": None,
                        "source": None,
                        "chunk_text": None,
                    },
                    "rank": None,
                    "source": None,
                    "chunk_text": None,
                }
            )
            continue

        retrieved_data = query_notes(collection, client, question)
        chunks = retrieved_data["texts"]
        metadatas = retrieved_data["metadatas"]

        matched = find_expected_evidence(expected_evidence, chunks, metadatas)
        if matched["matched"]:
            passed += 1

        case_results.append(
            {
                "id": case_id,
                "question": question,
                "answerable": True,
                "expected_evidence": expected_evidence,
                "evaluation": matched,
                "rank": matched["rank"],
                "source": matched["source"],
                "chunk_text": matched["chunk_text"],
            }
        )

    total_cases = len(evaluation_cases)
    answerable_cases = sum(1 for case in evaluation_cases if case["answerable"])
    unanswerable_cases = total_cases - answerable_cases
    failed = answerable_cases - passed
    accuracy = (passed / answerable_cases) * 100 if answerable_cases > 0 else 0.0

    return {
        "summary": {
            "total_cases": total_cases,
            "answerable_cases": answerable_cases,
            "unanswerable_cases": unanswerable_cases,
            "retrieval_passed": passed,
            "retrieval_failed": failed,
            "retrieval_accuracy": accuracy,
        },
        "results": case_results,
    }


def evaluate_abstention_thresholds(
    collection,
    client,
    evaluation_cases: list[dict],
    candidate_thresholds: list[float] | None = None,
) -> list[dict]:
    """Compare possible distance gates before choosing one for the RAG service."""
    if candidate_thresholds is None:
        candidate_thresholds = [0.15, 0.20, 0.22, 0.24, 0.25, 0.30, 0.50]

    case_distances = []
    for case in evaluation_cases:
        retrieved = query_notes(collection, client, case["question"])
        case_distances.append(
            {
                "id": case["id"],
                "answerable": case["answerable"],
                "distances": retrieved["distances"],
            }
        )

    sweep_results = []
    for threshold in candidate_thresholds:
        correct_generations = 0
        correct_abstentions = 0
        false_accepts = 0
        false_abstentions = 0

        for case in case_distances:
            abstained = should_abstain(case["distances"], threshold)
            if case["answerable"]:
                if not abstained:
                    correct_generations += 1
                else:
                    false_abstentions += 1
            else:
                if abstained:
                    correct_abstentions += 1
                else:
                    false_accepts += 1

        sweep_results.append(
            {
                "threshold": threshold,
                "correct_generations": correct_generations,
                "correct_abstentions": correct_abstentions,
                "false_accepts": false_accepts,
                "false_abstentions": false_abstentions,
            }
        )

    return sweep_results
