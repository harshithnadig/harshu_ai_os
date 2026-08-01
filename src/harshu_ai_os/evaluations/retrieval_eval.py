from harshu_ai_os.rag.chroma_store import get_notes_collection, query_notes
from harshu_ai_os.rag.embedding_client import get_embedding_client

evaluation_cases = [
    {
        "question": "What vector database does Harshu AI OS use?",
        "expected": "Chroma",
    },
    {
        "question": "Which API framework does Harshu AI OS use?",
        "expected": "FastAPI",
    },
    {
        "question": "How does Harshu AI OS handle simple user queries?",
        "expected": "router",
    },
    {
        "question": "What library validates request payloads in Harshu AI OS?",
        "expected": "Pydantic",
    },
    {
        "question": "What component supplies grounded context to the language model in Harshu AI OS?",
        "expected": "RAG",
    },
]


def evaluate_retrieval(expected, retrieved_chunks):
    """Returns True if expected text is found in any of the retrieved chunks, else False."""
    return any(expected in chunk for chunk in retrieved_chunks)


def run_retrieval_evaluation(collection, client, evaluation_cases):
    passed = 0
    case_results = []

    for case in evaluation_cases:
        question = case["question"]
        expected = case["expected"]

        # 1. Fetch chunks from Chroma
        retrieved_data = query_notes(collection, client, question)
        chunks = retrieved_data["texts"]
        metadatas = retrieved_data["metadatas"]

        # 2. Evaluate match
        matched = evaluate_retrieval_v2(expected, chunks, metadatas)
        if matched["matched"]:
            passed += 1

        # 3. Save result for this case
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


def evaluate_retrieval_v2(expected, chunks, metadatas):

    result = {
        "matched": False,
        "rank": None,
        "source": None,
        "chunk_text": None,
    }

    for rank, chunk in enumerate(chunks, start=1):

        if expected in chunk:

            result["matched"] = True
            result["rank"] = rank
            result["chunk_text"] = chunk
            result["source"] = metadatas[rank - 1]["source"]

            break

    return result


if __name__ == "__main__":
    client = get_embedding_client()
    collection = get_notes_collection()
    results = run_retrieval_evaluation(collection, client, evaluation_cases)

    # Print the full results
    print(results)

    # Print a nice summary with the % symbol
    summary = results["summary"]
    print(
        f"\nAccuracy: {summary['accuracy']:.2f}% ({summary['passed']}/{summary['total_cases']} passed)"
    )


