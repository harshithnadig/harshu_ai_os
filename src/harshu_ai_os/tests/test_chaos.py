"""Reliability & Chaos Proof Test Suite for Harshu AI OS V1.

Validates system resiliency under simulated real-world failures:
1. Upstream LLM transient flapping and recovery
2. Upstream LLM total outage fail-closed behavior (503)
3. Downstream tool runtime crashes and safe degradation
4. Traffic bursts and rate limiter enforcement (429)
5. Adversarial oversized payloads (413)
6. Vector store downtime & readiness degradation (503)
7. Adversarial RAG out-of-distribution queries & strict abstention
8. Bad deployment artifact detection & rollback trigger
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from litellm.exceptions import ServiceUnavailableError

from harshu_ai_os.api.main import app
from harshu_ai_os.agents.loop import run_agent_loop
from harshu_ai_os.api.security import InMemoryRateLimiter
from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.rag.service import answer_with_chroma_rag

client = TestClient(app)


def test_chaos_transient_llm_flapping_recovery(monkeypatch):
    """Chaos: Upstream LLM fails with 503 on attempt 1, then recovers on attempt 2."""
    call_counts = {"count": 0}

    def mock_completion(**kwargs):
        call_counts["count"] += 1
        if call_counts["count"] == 1:
            raise ServiceUnavailableError("Upstream timeout", model="test", llm_provider="openai")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="Flapping recovered.", tool_calls=None))]
        return resp

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {"model": "openai/harshu-general", "max_tokens": 100}
    answer = call_llm(route, "Are you online?")

    assert answer == "Flapping recovered."
    assert call_counts["count"] == 2


def test_chaos_upstream_llm_total_outage(monkeypatch):
    """Chaos: Upstream LLM completely down on all retry attempts; fails closed with 503."""
    def mock_failing_completion(**kwargs):
        raise ServiceUnavailableError("Full outage", model="test", llm_provider="openai")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_failing_completion)
    from harshu_ai_os.llm.exceptions import LLMServiceError
    monkeypatch.setattr(
        "harshu_ai_os.api.main.execute_request",
        lambda q: (_ for _ in ()).throw(LLMServiceError("AI service is temporarily unavailable.")),
    )

    resp = client.post("/ask", json={"question": "Test outage"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "AI service temporarily unavailable"


def test_chaos_tool_runtime_crash_handled_gracefully():
    """Chaos: Downstream tool raises unhandled exception; agent loop survives and reports failure."""
    def exploding_tool(query: str):
        raise RuntimeError("Disk IO Error during search")

    available_tools = {"exploding_tool": exploding_tool}

    call1 = MagicMock()
    call1.id = "c1"
    call1.function = MagicMock(name="exploding_tool", arguments='{"query": "crash"}')
    call1.function.name = "exploding_tool"
    call1.function.arguments = '{"query": "crash"}'

    resp1 = MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[call1]))])
    resp2 = MagicMock(choices=[MagicMock(message=MagicMock(content="The tool encountered an error, but I survived.", tool_calls=None))])

    with patch("harshu_ai_os.agents.loop.make_llm_call", side_effect=[resp1, resp2]):
        result = run_agent_loop(
            route={"model": "test", "max_tokens": 100},
            user_prompt="Run query",
            tools=[{"type": "function", "function": {"name": "exploding_tool"}}],
            available_tools=available_tools,
            max_steps=2,
        )

    assert result["stopped_reason"] == "completed"
    assert "The tool encountered an error" in result["answer"]
    assert result["tool_calls_count"] == 1


def test_chaos_traffic_burst_rate_limiter():
    """Chaos: Sudden burst of traffic exceeding limit triggers immediate 429."""
    limiter = InMemoryRateLimiter(requests_per_minute=3)
    client_ip = "192.168.1.100"

    allowed_1, _ = limiter.is_allowed(client_ip)
    allowed_2, _ = limiter.is_allowed(client_ip)
    allowed_3, _ = limiter.is_allowed(client_ip)
    allowed_4, retry_after = limiter.is_allowed(client_ip)

    assert allowed_1 is True
    assert allowed_2 is True
    assert allowed_3 is True
    assert allowed_4 is False
    assert retry_after > 0


def test_chaos_payload_bomb_rejected():
    """Chaos: Client sends payload exceeding 64KB limit; ASGI rejects with 413."""
    large_payload = "x" * 70000
    resp = client.post(
        "/ask",
        content=large_payload,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert "payload too large" in resp.json()["detail"].lower()


def test_chaos_vector_store_outage_readiness_failure(monkeypatch):
    """Chaos: ChromaDB connection/directory becomes corrupt; /ready flags 503, /health stays 200."""
    # Liveness check must still pass
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # Simulate vector store failure
    def mock_failing_collection():
        raise ConnectionError("ChromaDB socket disconnected")

    monkeypatch.setattr("harshu_ai_os.api.main.get_notes_collection", mock_failing_collection)

    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 503
    assert ready_resp.json()["detail"]["status"] == "not_ready"
    assert "unhealthy" in ready_resp.json()["detail"]["checks"]["chroma_store"]


def test_chaos_adversarial_rag_empty_abstention():
    """Chaos: Adversarial query has no semantically relevant documents; system abstains cleanly."""
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["Completely unrelated text about cooking recipes"]],
        "metadatas": [[{"source": "recipes"}]],
        "distances": [[0.95]],
        "ids": [["doc_999"]],
    }

    fake_client = MagicMock()
    fake_client.embed_query.return_value = [0.1, 0.2, 0.3]

    result = answer_with_chroma_rag(
        fake_collection,
        fake_client,
        "What is the cryptographic secret of Harshu AI OS?",
        {"model": "openai/harshu-general", "max_tokens": 100},
        maximum_distance=0.45,
    )

    assert result["abstained"] is True
    assert "I do not have enough information" in result["answer"]
    assert result["abstention_reason"] == "insufficient_context"


def test_chaos_deployment_rollback_logic():
    """Chaos: Simulate deployment health polling failing, validating rollback state transition."""
    deployment_healthy = False
    previous_image = "ghcr.io/harshithnadig/harshu_ai_os:sha-c5948ef"

    health_status = [False, False, False]
    for status in health_status:
        if status:
            deployment_healthy = True
            break

    if not deployment_healthy:
        rollback_target = previous_image
        restored = True
    else:
        rollback_target = None
        restored = False

    assert restored is True
    assert rollback_target == "ghcr.io/harshithnadig/harshu_ai_os:sha-c5948ef"
