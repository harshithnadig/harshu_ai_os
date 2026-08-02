# This file stores our evaluation test cases.
# We keep them separate so it's easy to add or modify test data without touching the logic.

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
