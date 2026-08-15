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


def fake_call_llm(route: dict, user_prompt: str, **kwargs):
    return {
        "answer": "fake answer",
        "tool_used": False,
        "tool_name": None,
        "tool_query": None,
        "tool_sources": [],
    }


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
    assert data["tool_used"] is False
    assert data["tool_name"] is None
    assert data["tool_query"] is None
    assert data["tool_sources"] == []


def test_ask_endpoint_with_web_search_tool(monkeypatch):
    def fake_call_llm_tool(route: dict, user_prompt: str, **kwargs):
        return {
            "answer": "Python 3.14 includes new features.",
            "tool_used": True,
            "tool_name": "web_search",
            "tool_query": "Python 3.14 release",
            "tool_sources": [
                {"title": "Python 3.14 News", "url": "https://python.org/3.14"}
            ],
        }

    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model", fake_classify_task
    )
    monkeypatch.setattr("harshu_ai_os.api.main.call_llm", fake_call_llm_tool)

    response = client.post("/ask", json={"question": "What is Python 3.14?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Python 3.14 includes new features."
    assert data["tool_used"] is True
    assert data["tool_name"] == "web_search"
    assert data["tool_query"] == "Python 3.14 release"
    assert data["tool_sources"] == [
        {"title": "Python 3.14 News", "url": "https://python.org/3.14"}
    ]


def test_ask_endpoint_llm_failure(monkeypatch):
    def fake_classify_task(question):
        return TaskClassification(
            complexity="general",
            needs_current_information=False,
            needs_tool=False,
        )

    def fake_call_llm(route, question, **kwargs):
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


def fake_answer_with_chroma_rag(
    collection,
    embedding_client,
    question,
    route,
    maximum_distance,
):
    assert question == "What does ChromaDB do?"
    assert collection is not None
    assert embedding_client is not None
    assert route["model"] == "openai/harshu-general"
    assert maximum_distance == 0.5
    return {
        "answer": "ChromaDB retrieves relevant stored notes.",
        "abstained": False,
        "abstention_reason": None,
        "judge_reason": "Context directly supports.",
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
        "retrieval_ms": 12.5,
        "judge_ms": 45.0,
        "generation_ms": 80.2,
        "total_ms": 137.7,
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
    assert data["model"] == "openai/harshu-general"
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
    assert data["abstained"] is False
    assert data["abstention_reason"] is None
    assert data["judge_reason"] == "Context directly supports."
    assert data["retrieval_ms"] == 12.5
    assert data["judge_ms"] == 45.0
    assert data["generation_ms"] == 80.2
    assert data["total_ms"] == 137.7
