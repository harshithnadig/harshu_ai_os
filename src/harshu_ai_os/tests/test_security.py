"""Tests for Security Baseline: auth fail-closed, rate limiting, payload limits, and readiness."""

from fastapi.testclient import TestClient
from harshu_ai_os.api.main import app
from harshu_ai_os.api.security import InMemoryRateLimiter, global_rate_limiter, is_auth_disabled

client = TestClient(app)


def test_health_and_readiness_probes_remain_public(monkeypatch):
    """Probes must remain public even when auth is unconfigured or failing closed."""
    monkeypatch.delenv("HARSHU_API_KEY", raising=False)
    monkeypatch.delenv("HARSHU_AUTH_DISABLED", raising=False)

    # Liveness must return 200 without credentials
    resp_liveness = client.get("/health")
    assert resp_liveness.status_code == 200
    assert resp_liveness.json() == {"status": "healthy"}

    # Readiness must return 200 without credentials
    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    assert resp_ready.json()["status"] == "ready"


def test_readiness_probe_failure_mode(monkeypatch):
    """Readiness fails with 503 when vector store is down, but not due to auth."""
    def mock_failing_collection():
        raise RuntimeError("Disk full / DB unavailable")

    monkeypatch.setattr("harshu_ai_os.api.main.get_notes_collection", mock_failing_collection)
    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.json()["detail"]
    assert data["status"] == "not_ready"
    assert "unhealthy: RuntimeError" in data["checks"]["chroma_store"]


def test_state_1_no_key_no_bypass_fails_closed(monkeypatch):
    """State 1: No HARSHU_API_KEY and no HARSHU_AUTH_DISABLED fails closed with 503."""
    monkeypatch.delenv("HARSHU_API_KEY", raising=False)
    monkeypatch.delenv("HARSHU_AUTH_DISABLED", raising=False)

    for endpoint in ("/ask", "/ask/rag", "/ask/agent"):
        resp = client.post(endpoint, json={"question": "What is Harshu AI OS?"})
        assert resp.status_code == 503
        assert "server authentication is unconfigured" in resp.json()["detail"]


def test_state_2_explicit_auth_disabled_permitted(monkeypatch):
    """State 2: No HARSHU_API_KEY but HARSHU_AUTH_DISABLED=true permits request past auth."""
    monkeypatch.delenv("HARSHU_API_KEY", raising=False)
    monkeypatch.setenv("HARSHU_AUTH_DISABLED", "true")

    # Mock execution so it does not require a live provider
    monkeypatch.setattr(
        "harshu_ai_os.api.main.execute_request",
        lambda q: {"answer": "Mocked response", "complexity": "simple", "workflow_used": "direct"},
    )

    resp = client.post("/ask", json={"question": "What is Harshu AI OS?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Mocked response"


def test_state_3_configured_key_with_missing_or_wrong_credential_returns_401(monkeypatch):
    """State 3: HARSHU_API_KEY configured + missing/wrong client credential -> HTTP 401."""
    test_key = "secret-test-key-998877"
    monkeypatch.setenv("HARSHU_API_KEY", test_key)
    monkeypatch.delenv("HARSHU_AUTH_DISABLED", raising=False)

    # 1. Missing header -> 401
    resp_missing = client.post("/ask", json={"question": "Test query"})
    assert resp_missing.status_code == 401
    assert resp_missing.json()["detail"] == "Invalid or missing API key"

    # 2. Invalid X-API-Key -> 401
    resp_invalid_key = client.post(
        "/ask",
        json={"question": "Test query"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp_invalid_key.status_code == 401

    # 3. Invalid Authorization Bearer -> 401
    resp_invalid_bearer = client.post(
        "/ask",
        json={"question": "Test query"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp_invalid_bearer.status_code == 401


def test_state_4_configured_key_with_correct_credential_permitted(monkeypatch):
    """State 4: HARSHU_API_KEY configured + correct client credential -> request permitted."""
    test_key = "secret-test-key-998877"
    monkeypatch.setenv("HARSHU_API_KEY", test_key)
    monkeypatch.delenv("HARSHU_AUTH_DISABLED", raising=False)

    monkeypatch.setattr(
        "harshu_ai_os.api.main.execute_request",
        lambda q: {"answer": "Authenticated response", "complexity": "simple", "workflow_used": "direct"},
    )

    # Valid X-API-Key
    resp_header = client.post(
        "/ask",
        json={"question": "Test query"},
        headers={"X-API-Key": test_key},
    )
    assert resp_header.status_code == 200
    assert resp_header.json()["answer"] == "Authenticated response"

    # Valid Authorization: Bearer <key>
    resp_bearer = client.post(
        "/ask",
        json={"question": "Test query"},
        headers={"Authorization": f"Bearer {test_key}"},
    )
    assert resp_bearer.status_code == 200
    assert resp_bearer.json()["answer"] == "Authenticated response"


def test_auth_disabled_rejects_false_and_invalid_values(monkeypatch):
    """HARSHU_AUTH_DISABLED values that are not '1', 'true', 'yes' must NOT disable auth."""
    monkeypatch.delenv("HARSHU_API_KEY", raising=False)

    for invalid_val in ("false", "0", "no", "random-string", "none"):
        monkeypatch.setenv("HARSHU_AUTH_DISABLED", invalid_val)
        assert is_auth_disabled() is False

        resp = client.post("/ask", json={"question": "Test query"})
        assert resp.status_code == 503


def test_auth_disabled_accepts_truthy_variations(monkeypatch):
    """Accepts case-insensitive '1', 'true', 'yes'."""
    for truthy_val in ("1", "true", "TRUE", "True", "yes", "YES"):
        monkeypatch.setenv("HARSHU_AUTH_DISABLED", truthy_val)
        assert is_auth_disabled() is True


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
    monkeypatch.setenv("HARSHU_AUTH_DISABLED", "true")

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


def test_question_max_length_validation(monkeypatch):
    monkeypatch.setenv("HARSHU_AUTH_DISABLED", "true")
    overly_long_question = "A" * 4001
    resp = client.post("/ask", json={"question": overly_long_question})
    assert resp.status_code == 422
