"""Tests for Security Baseline: auth, rate limiting, payload limits, and readiness."""

from fastapi.testclient import TestClient
from harshu_ai_os.api.main import app
from harshu_ai_os.api.security import InMemoryRateLimiter, global_rate_limiter

client = TestClient(app)


def test_health_and_readiness_probes():
    # Liveness
    resp_liveness = client.get("/health")
    assert resp_liveness.status_code == 200
    assert resp_liveness.json() == {"status": "healthy"}

    # Readiness
    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    data = resp_ready.json()
    assert data["status"] == "ready"
    assert data["checks"]["chroma_store"] == "ok"


def test_readiness_probe_failure_mode(monkeypatch):
    def mock_failing_collection():
        raise RuntimeError("Disk full / DB unavailable")

    monkeypatch.setattr("harshu_ai_os.api.main.get_notes_collection", mock_failing_collection)
    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.json()["detail"]
    assert data["status"] == "not_ready"
    assert "unhealthy: RuntimeError" in data["checks"]["chroma_store"]


def test_auth_dev_mode_allowed(monkeypatch):
    monkeypatch.delenv("HARSHU_API_KEY", raising=False)
    resp = client.post("/ask", json={"question": "What is Harshu AI OS?"})
    # Should not be rejected with 401
    assert resp.status_code != 401


def test_auth_enforced_when_configured(monkeypatch):
    test_key = "secret-test-key-998877"
    monkeypatch.setenv("HARSHU_API_KEY", test_key)

    # 1. Missing key -> 401
    resp_missing = client.post("/ask", json={"question": "Test query"})
    assert resp_missing.status_code == 401
    assert resp_missing.json()["detail"] == "Invalid or missing API key"

    # 2. Invalid key -> 401
    resp_invalid = client.post(
        "/ask",
        json={"question": "Test query"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp_invalid.status_code == 401

    # 3. Valid X-API-Key -> authorized
    resp_valid_header = client.post(
        "/ask",
        json={"question": "Test query"},
        headers={"X-API-Key": test_key},
    )
    assert resp_valid_header.status_code != 401

    # 4. Valid Authorization: Bearer <key> -> authorized
    resp_valid_bearer = client.post(
        "/ask",
        json={"question": "Test query"},
        headers={"Authorization": f"Bearer {test_key}"},
    )
    assert resp_valid_bearer.status_code != 401


def test_rate_limiter_unit_behavior():
    limiter = InMemoryRateLimiter(requests_per_minute=3)
    client_id = "test-client-1"

    t0 = 1000.0
    allowed, retry = limiter.is_allowed(client_id, now=t0)
    assert allowed is True
    assert retry == 0

    allowed, retry = limiter.is_allowed(client_id, now=t0 + 1.0)
    assert allowed is True

    allowed, retry = limiter.is_allowed(client_id, now=t0 + 2.0)
    assert allowed is True

    # 4th request within 60s window must be rejected
    allowed, retry = limiter.is_allowed(client_id, now=t0 + 3.0)
    assert allowed is False
    assert retry == 57

    # Different client must still be allowed
    allowed_other, _ = limiter.is_allowed("test-client-2", now=t0 + 3.0)
    assert allowed_other is True

    # After window passes, original client allowed again
    allowed_after, _ = limiter.is_allowed(client_id, now=t0 + 61.0)
    assert allowed_after is True


def test_rate_limiting_endpoint_integration(monkeypatch):
    global_rate_limiter.reset()
    monkeypatch.setattr(global_rate_limiter, "requests_per_minute", 2)

    try:
        r1 = client.post("/ask", json={"question": "First"})
        assert r1.status_code != 429

        r2 = client.post("/ask", json={"question": "Second"})
        assert r2.status_code != 429

        # Third must trigger 429
        r3 = client.post("/ask", json={"question": "Third"})
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]
        assert "Retry-After" in r3.headers
    finally:
        global_rate_limiter.reset()
        monkeypatch.setattr(global_rate_limiter, "requests_per_minute", 60)


def test_payload_size_limit_rejected():
    large_payload = "x" * 70000
    headers = {"Content-Length": str(len(large_payload)), "Content-Type": "application/json"}
    resp = client.post("/ask", content=large_payload, headers=headers)
    assert resp.status_code == 413
    assert resp.json()["detail"] == "Request payload too large (max 64KB)"
    assert "x-request-id" in resp.headers


def test_question_max_length_validation():
    overly_long_question = "A" * 4001
    resp = client.post("/ask", json={"question": overly_long_question})
    assert resp.status_code == 422
