from fastapi.testclient import TestClient

from harshu_ai_os.api.main import app
from harshu_ai_os.llm.router import TaskClassification
from harshu_ai_os.llm.exceptions import LLMServiceError


client = TestClient(app)


def fake_classify_task(question: str):
    return TaskClassification(
        complexity="general",
        needs_current_information=False,
        needs_tool=False,
    )


def fake_call_llm(route: dict, user_prompt: str):
    return "fake answer"


def test_ask_endpoint(monkeypatch):

    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model", fake_classify_task
    )

    monkeypatch.setattr("harshu_ai_os.api.main.call_llm", fake_call_llm)

    response = client.post("/ask", json={"question": "Explain RAG"})

    assert response.status_code == 200

    data = response.json()

    assert data["complexity"] == "general"
    assert data["answer"] == "fake answer"


def test_ask_endpoint_llm_failure(monkeypatch):

    def fake_classify_task(question):
        return TaskClassification(
            complexity="general",
            needs_current_information=False,
            needs_tool=False,
        )

    def fake_call_llm(route, question):
        raise LLMServiceError("provider unavailable")

    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model",
        fake_classify_task,
    )

    monkeypatch.setattr(
        "harshu_ai_os.api.main.call_llm",
        fake_call_llm,
    )

    response = client.post(
        "/ask",
        json={"question": "Explain RAG"},
    )

    assert response.status_code == 503


def test_ask_endpoint_classifier_failure(monkeypatch):

    def fake_classify_task(question):
        raise LLMServiceError("classifier unavailable")

    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model",
        fake_classify_task,
    )

    response = client.post(
        "/ask",
        json={"question": "Explain RAG"},
    )

    assert response.status_code == 503

def fake_get_notes_collection():
    return object()


def fake_get_embedding_client():
    return object()


def fake_create_rag_generator(route):
    return object()


def fake_answer_with_chroma_rag(
    collection,
    embedding_client,
    question,
    generate_text,
):
    assert question == "What does ChromaDB do?"
    assert collection is not None
    assert embedding_client is not None
    assert generate_text is not None
    return {
        "answer": "ChromaDB retrieves relevant stored notes.",
        "context": "ChromaDB stores embeddings and retrieves notes.",
        "distances": [0.2],
        "ids": ["note-2"],
        "metadatas": [
        {
            "source": "manual",
            "position": 2,
        }
    ],
    "citations": [
        {
            "source": "manual",
            "chunk_id": "note-2",
            "chunk_index": None,
            "distance": 0.2,
        }
    ],
}

def test_ask_rag_endpoint_returns_grounded_response(monkeypatch):
    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model",
        fake_classify_task,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.get_notes_collection",
        fake_get_notes_collection,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.get_embedding_client",
        fake_get_embedding_client,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.create_rag_generator",
        fake_create_rag_generator,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.answer_with_chroma_rag",
        fake_answer_with_chroma_rag,
    )

    response = client.post(
        "/ask/rag",
        json={"question": "What does ChromaDB do?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "ChromaDB retrieves relevant stored notes."
    assert data["complexity"] == "general"
    assert data["model"] == "gemini/gemini-2.5-flash"
    assert data["ids"] == ["note-2"]
    assert data["distances"] == [0.2]
    assert data["metadatas"][0]["position"] == 2
    assert "ChromaDB stores embeddings" in data["context"]
    assert data["citations"] == [
    {
        "source": "manual",
        "chunk_id": "note-2",
        "chunk_index": None,
        "distance": 0.2,
    }
]