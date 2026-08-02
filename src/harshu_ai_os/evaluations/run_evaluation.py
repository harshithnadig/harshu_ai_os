# This is the entry point script to execute the evaluation. 
# Run this script directly from the terminal.

from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client

# Import from our newly split modules
from harshu_ai_os.evaluations.retrieval_cases import evaluation_cases
from harshu_ai_os.evaluations.retrieval_evaluator import run_retrieval_evaluation
from harshu_ai_os.evaluations.retrieval_metrics import calculate_hit_at_k

if __name__ == "__main__":
    # Initialize the client and collection
    client = get_embedding_client()
    collection = get_notes_collection()
    
    # Run the evaluation logic
    results = run_retrieval_evaluation(collection, client, evaluation_cases)

    # Print the full detailed results dictionary
    print(results)

    # Extract and print a nicely formatted summary percentage
    summary = results["summary"]
    print(
        f"\nAccuracy: {summary['accuracy']:.2f}% ({summary['passed']}/{summary['total_cases']} passed)"
    )

    # Example tests for the hit_at_k metric function
    print(calculate_hit_at_k(2, 5))
    print(calculate_hit_at_k(6, 5))
    print(calculate_hit_at_k(None, 5))
