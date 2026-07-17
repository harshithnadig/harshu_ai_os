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
        "harshu_ai_os.api.main.classify_task_with_model",
        fake_classify_task
    )

    monkeypatch.setattr(
        "harshu_ai_os.api.main.call_llm",
        fake_call_llm
    )

    response = client.post(
        "/ask",
        json={
            "question": "Explain RAG"
        }
    )

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
        json={
            "question": "Explain RAG"
        },
    )

    assert response.status_code == 503