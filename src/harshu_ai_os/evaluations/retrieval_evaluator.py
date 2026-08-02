# This file contains the core logic for running the evaluation against our Retrieval-Augmented Generation (RAG) system.

from harshu_ai_os.rag.chroma_store import query_notes


def evaluate_retrieval(expected, retrieved_chunks):
    """Returns True if expected text is found in any of the retrieved chunks, else False."""
    return any(expected in chunk for chunk in retrieved_chunks)


def evaluate_retrieval_v2(expected, chunks, metadatas):
    """
    Evaluates the retrieved chunks by finding the exact rank of the expected string.
    Returns a dictionary containing matched status, rank, source metadata, and chunk text.
    """
    result = {
        "matched": False,
        "rank": None,
        "source": None,
        "chunk_text": None,
    }

    # enumerate(chunks, start=1) starts counting from 1 instead of 0
    for rank, chunk in enumerate(chunks, start=1):

        if expected in chunk:

            result["matched"] = True
            result["rank"] = rank
            result["chunk_text"] = chunk
            result["source"] = metadatas[rank - 1]["source"]

            break

    return result


def run_retrieval_evaluation(collection, client, evaluation_cases):
    """
    Runs evaluation for a list of test cases, fetching relevant data from Chroma 
    and then calculating accuracy. Returns a dictionary with summary and detailed results.
    """
    passed = 0
    case_results = []

    for case in evaluation_cases:
        question = case["question"]
        expected = case["expected"]

        # 1. Fetch chunks from Chroma database
        retrieved_data = query_notes(collection, client, question)
        chunks = retrieved_data["texts"]
        metadatas = retrieved_data["metadatas"]

        # 2. Evaluate match by checking if expected text is in retrieved chunks
        matched = evaluate_retrieval_v2(expected, chunks, metadatas)
        if matched["matched"]:
            passed += 1

        # 3. Save result for this particular test case
        case_results.append(
            {
                "question": question,
                "expected": expected,
                "evaluation": matched,
                "rank": matched["rank"],
                "source": matched["source"],
                "chunk_text": matched["chunk_text"],
            }
        )

    # Calculate overall summary metrics
    total_cases = len(evaluation_cases)
    failed = total_cases - passed
    accuracy = (passed / total_cases) * 100 if total_cases > 0 else 0.0

    return {
        "summary": {
            "total_cases": total_cases,
            "passed": passed,
            "failed": failed,
            "accuracy": accuracy,
        },
        "results": case_results,
    }
