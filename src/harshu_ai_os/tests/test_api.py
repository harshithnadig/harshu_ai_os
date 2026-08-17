from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from harshu_ai_os.api.main import app
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.llm.router import TaskClassification
from harshu_ai_os.orchestrator.service import RequestPlan

client = TestClient(app)


def fake_classify_task(question: str):
    return TaskClassification(
        complexity="general",
        needs_current_information=False,
        needs_tool=False,
    )


# ======================================================================
# UNIFIED /ask ORCHESTRATOR TESTS
# ======================================================================

def test_ask_endpoint_direct(monkeypatch):
    """Test /ask routes ordinary questions to DIRECT workflow."""
    def fake_execute(question, **kwargs):
        return {
            "answer": "A list comprehension is a concise way to create lists.",
            "complexity": "general",
            "workflow_used": "direct",
            "model": "openai/harshu-general",
            "tool_used": False,
            "tool_calls_count": 0,
            "tool_sources": [],
            "citations": [],
            "abstained": False,
            "stopped_reason": "direct_answer",
            "steps_taken": 0,
        }

    monkeypatch.setattr("harshu_ai_os.api.main.execute_request", fake_execute)

    response = client.post("/ask", json={"question": "Explain Python list comprehensions."})

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_used"] == "direct"
    assert data["complexity"] == "general"
    assert data["model"] == "openai/harshu-general"
    assert data["answer"] == "A list comprehension is a concise way to create lists."
    assert data["tool_used"] is False
    assert data["tool_calls_count"] == 0
    assert data["citations"] == []
    assert data["abstained"] is False


def test_ask_endpoint_agent_current_information(monkeypatch):
    """Test /ask routes current-information questions to AGENT workflow."""
    def fake_execute(question, **kwargs):
        return {
            "answer": "The latest stable Python release is Python 3.14.7.",
            "complexity": "general",
            "workflow_used": "agent",
            "model": "openai/harshu-general",
            "tool_used": True,
            "tool_calls_count": 2,
            "tool_sources": [
                {"title": "Python Downloads", "url": "https://python.org/downloads"}
            ],
            "citations": [],
            "abstained": False,
            "stopped_reason": "completed",
            "steps_taken": 2,
        }

    monkeypatch.setattr("harshu_ai_os.api.main.execute_request", fake_execute)

    response = client.post("/ask", json={"question": "What is the latest stable Python release right now?"})

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_used"] == "agent"
    assert data["tool_used"] is True
    assert data["tool_calls_count"] == 2
    assert len(data["tool_sources"]) == 1
    assert data["stopped_reason"] == "completed"


def test_ask_endpoint_agent_internal_knowledge(monkeypatch):
    """Test /ask routes internal project lookup questions to AGENT workflow."""
    def fake_execute(question, **kwargs):
        return {
            "answer": "Harshu AI OS query router classifies into simple, general, and complex.",
            "complexity": "general",
            "workflow_used": "agent",
            "model": "openai/harshu-general",
            "tool_used": True,
            "tool_calls_count": 1,
            "tool_sources": [
                {"title": "harshu_ai_os_details.txt", "url": ""}
            ],
            "citations": [],
            "abstained": False,
            "stopped_reason": "completed",
            "steps_taken": 1,
        }

    monkeypatch.setattr("harshu_ai_os.api.main.execute_request", fake_execute)

    response = client.post("/ask", json={"question": "What categories does Harshu AI OS router use?"})

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_used"] == "agent"
    assert data["tool_used"] is True
    assert data["tool_calls_count"] == 1
    assert data["tool_sources"][0]["title"] == "harshu_ai_os_details.txt"


def test_ask_endpoint_agent_mixed(monkeypatch):
    """Test /ask routes mixed internal+web questions to AGENT workflow."""
    def fake_execute(question, **kwargs):
        return {
            "answer": "Gemini Flash is used for simple queries, and it was created by Google.",
            "complexity": "complex",
            "workflow_used": "agent",
            "model": "openai/harshu-reasoning",
            "tool_used": True,
            "tool_calls_count": 3,
            "tool_sources": [
                {"title": "harshu_ai_os_details.txt", "url": ""},
                {"title": "Google DeepMind Gemini", "url": "https://deepmind.google/gemini"},
            ],
            "citations": [],
            "abstained": False,
            "stopped_reason": "completed",
            "steps_taken": 3,
        }

    monkeypatch.setattr("harshu_ai_os.api.main.execute_request", fake_execute)

    response = client.post(
        "/ask",
        json={"question": "Which lightweight model is used in Harshu AI OS and who created it?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_used"] == "agent"
    assert data["tool_used"] is True
    assert data["tool_calls_count"] == 3
    assert len(data["tool_sources"]) == 2


def test_ask_endpoint_strict_rag(monkeypatch):
    """Test /ask routes explicit project-document grounding questions to STRICT_RAG workflow."""
    def fake_execute(question, **kwargs):
        return {
            "answer": "According to indexed documents, FastAPI exposes the API endpoints.",
            "complexity": "general",
            "workflow_used": "strict_rag",
            "model": "openai/harshu-general",
            "tool_used": False,
            "tool_calls_count": 0,
            "tool_sources": [],
            "citations": [
                {
                    "source": "harshu_ai_os_overview.txt",
                    "chunk_id": "harshu_ai_os_overview-0",
                    "chunk_index": 0,
                    "distance": 0.12,
                }
            ],
            "abstained": False,
            "abstention_reason": None,
            "judge_reason": "Direct evidence present.",
            "stopped_reason": "rag_grounded",
            "steps_taken": 0,
        }

    monkeypatch.setattr("harshu_ai_os.api.main.execute_request", fake_execute)

    response = client.post(
        "/ask",
        json={"question": "According to indexed project documents, what exposes API endpoints?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_used"] == "strict_rag"
    assert data["abstained"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source"] == "harshu_ai_os_overview.txt"


def test_ask_endpoint_llm_failure(monkeypatch):
    """Test /ask handles LLMServiceError with 503 status code."""
    def fake_execute_fail(question, **kwargs):
        raise LLMServiceError("AI provider unreachable")

    monkeypatch.setattr("harshu_ai_os.api.main.execute_request", fake_execute_fail)

    response = client.post("/ask", json={"question": "Explain Python"})

    assert response.status_code == 503
    assert response.json()["detail"] == "AI service temporarily unavailable"


def test_ask_endpoint_planner_failure(monkeypatch):
    """Test /ask maps real planner failure (e.g. malformed JSON from classifier) to 503."""
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "Malformed response"
    fake_response.choices = [fake_choice]

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.completion",
        lambda **kwargs: fake_response,
    )

    response = client.post("/ask", json={"question": "What is the latest release?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "AI service temporarily unavailable"


def test_ask_endpoint_empty_question():
    """Test /ask handles empty question with 400 status code."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


# ======================================================================
# SPECIALIZED DIAGNOSTIC ENDPOINTS (/ask/rag and /ask/agent)
# ======================================================================

def test_ask_rag_endpoint_success(monkeypatch):
    fake_collection = MagicMock()
    fake_embedding_client = MagicMock()

    def fake_rag(collection, client, question, route, maximum_distance):
        assert route["model"] == "openai/harshu-general"
        assert question == "What is ChromaDB?"
        return {
            "answer": "ChromaDB is a vector database.",
            "abstained": False,
            "abstention_reason": None,
            "judge_reason": "Context was sufficient.",
            "context": "ChromaDB stores document embeddings.",
            "distances": [0.12],
            "ids": ["doc_0_chunk_1"],
            "metadatas": [{"source": "docs.txt", "chunk_index": 1}],
            "citations": [
                {
                    "source": "docs.txt",
                    "chunk_id": "doc_0_chunk_1",
                    "chunk_index": 1,
                    "distance": 0.12,
                }
            ],
            "retrieval_ms": 12.5,
            "judge_ms": 45.0,
            "generation_ms": 80.2,
            "total_ms": 137.7,
        }

    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model",
        fake_classify_task,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.get_notes_collection",
        lambda: fake_collection,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.get_embedding_client",
        lambda: fake_embedding_client,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.answer_with_chroma_rag",
        fake_rag,
    )

    response = client.post(
        "/ask/rag",
        json={"question": "What is ChromaDB?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "ChromaDB is a vector database."
    assert data["complexity"] == "general"
    assert data["model"] == "openai/harshu-general"
    assert data["abstained"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "doc_0_chunk_1"


def test_ask_agent_endpoint_success(monkeypatch):
    def fake_run_agent_loop(route: dict, user_prompt: str, tools: list, available_tools: dict, **kwargs):
        assert route["model"] == "openai/harshu-general"
        assert user_prompt == "What is Python 3.14?"
        assert len(tools) == 2
        tool_names = [t["function"]["name"] for t in tools]
        assert tool_names == ["web_search", "rag_lookup"]
        return {
            "answer": "Python 3.14 was released with template strings and performance updates.",
            "steps_taken": 2,
            "tool_calls_count": 2,
            "tool_sources": [
                {"title": "Python 3.14 Docs", "url": "https://docs.python.org/3.14"},
                {"title": "Python Releases", "url": "https://python.org/downloads"},
            ],
            "stopped_reason": "completed",
            "tool_used": True,
        }

    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model",
        fake_classify_task,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.run_agent_loop",
        fake_run_agent_loop,
    )

    response = client.post(
        "/ask/agent",
        json={"question": "What is Python 3.14?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Python 3.14 was released with template strings and performance updates."
    assert data["complexity"] == "general"
    assert data["model"] == "openai/harshu-general"
    assert data["steps_taken"] == 2
    assert data["tool_calls_count"] == 2
    assert data["stopped_reason"] == "completed"
    assert data["tool_used"] is True
    assert len(data["tool_sources"]) == 2


def test_ask_agent_endpoint_direct_answer(monkeypatch):
    def fake_run_agent_loop_direct(route: dict, user_prompt: str, tools: list, available_tools: dict, **kwargs):
        return {
            "answer": "Paris is the capital of France.",
            "steps_taken": 0,
            "tool_calls_count": 0,
            "tool_sources": [],
            "stopped_reason": "direct_answer",
            "tool_used": False,
        }

    monkeypatch.setattr(
        "harshu_ai_os.api.main.classify_task_with_model",
        fake_classify_task,
    )
    monkeypatch.setattr(
        "harshu_ai_os.api.main.run_agent_loop",
        fake_run_agent_loop_direct,
    )

    response = client.post(
        "/ask/agent",
        json={"question": "What is the capital of France?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Paris is the capital of France."
    assert data["steps_taken"] == 0
    assert data["tool_used"] is False


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
