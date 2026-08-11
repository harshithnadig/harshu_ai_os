"""Run retrieval cases and record where the expected evidence appeared."""

from harshu_ai_os.rag.chroma_store import query_notes
from harshu_ai_os.rag.service import should_abstain


import time

def find_expected_evidence(expected, expected_ids, chunks, metadatas, retrieved_ids):
    """Return the first 1-based rank containing the expected text or chunk ID."""
    for rank, (chunk, metadata, chunk_id) in enumerate(zip(chunks, metadatas, retrieved_ids), start=1):
        if expected_ids:
            if chunk_id in expected_ids:
                return {
                    "matched": True,
                    "rank": rank,
                    "source": metadata["source"],
                    "chunk_text": chunk,
                }
        elif expected and expected in chunk:
            return {
                "matched": True,
                "rank": rank,
                "source": metadata["source"],
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
                    "retrieved_ids": [],
                    "latency": 0.0,
                }
            )
            continue

        start_time = time.perf_counter()
        retrieved_data = query_notes(collection, client, question)
        latency = time.perf_counter() - start_time
        
        chunks = retrieved_data["texts"]
        metadatas = retrieved_data["metadatas"]
        retrieved_ids = retrieved_data["ids"]
        expected_ids = case.get("expected_chunk_ids", [])

        matched = find_expected_evidence(expected_evidence, expected_ids, chunks, metadatas, retrieved_ids)
        if matched["matched"]:
            passed += 1

        case_results.append(
            {
                "id": case_id,
                "category": case.get("category", "unknown"),
                "question": question,
                "answerable": True,
                "expected_evidence": expected_evidence,
                "expected_chunk_ids": expected_ids,
                "evaluation": matched,
                "rank": matched["rank"],
                "source": matched["source"],
                "chunk_text": matched["chunk_text"],
                "retrieved_ids": retrieved_ids,
                "latency": latency,
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
