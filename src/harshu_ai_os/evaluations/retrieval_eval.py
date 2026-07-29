evaluation_cases = [
    {
        "question": "What vector database does Harshu AI OS use?",
        "expected": "Chroma",
    },
    {
        "question": "Which API framework does Harshu AI OS use?",
        "expected": "FastAPI",
    },
]
# Example retrieved chunks for testing evaluator behaviour
retrieved_chunks = [
    "Harshu AI OS uses Chroma as vector database.",
    "FastAPI is used for API development."
]

def evaluate_retrieval(evaluation_cases, retrieved_chunks):
    results = {}
    for case in evaluation_cases:
        matched = any(case["expected"] in chunk for chunk in retrieved_chunks)
        results[case["question"]] = {
            "expected": case["expected"],
            "matched": matched
        }
    return results



if __name__ == "__main__":
    print(evaluate_retrieval(evaluation_cases,retrieved_chunks))


    
        
        
    
