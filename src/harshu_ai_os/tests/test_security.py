"""Tests for Security Baseline: auth fail-closed, rate limiting, payload limits, and readiness."""

import asyncio
import logging
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


# ======================================================================
# Actual Byte Body-Limit Enforcement Tests (Pure ASGI Harness)
# ======================================================================

async def _raw_asgi_post(app_instance, headers=None, chunks=None, path="/ask"):
    """Focused raw ASGI test harness to construct requests TestClient cannot accurately build."""
    sent_messages = []
    chunk_iter = iter(chunks or [])

    async def receive():
        try:
            return next(chunk_iter)
        except StopIteration:
            return {"type": "http.disconnect"}

    async def send(message):
        sent_messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
    }

    await app_instance(scope, receive, send)

    start_msg = next((m for m in sent_messages if m["type"] == "http.response.start"), None)
    body_msg = next((m for m in sent_messages if m["type"] == "http.response.body"), None)
    status = start_msg["status"] if start_msg else None
    resp_headers = dict(start_msg.get("headers", [])) if start_msg else {}
    body = body_msg.get("body", b"") if body_msg else b""
    return status, resp_headers, body


def test_declared_content_length_greater_than_64kb_immediate_413():
    """Declared Content-Length > 64 KiB triggers immediate 413 with X-Request-ID."""
    custom_req_id = "cl-too-large-id-01"
    headers = [
        (b"content-length", b"70000"),
        (b"content-type", b"application/json"),
        (b"x-request-id", custom_req_id.encode("latin1")),
    ]
    status, resp_headers, body = asyncio.run(
        _raw_asgi_post(app, headers=headers, chunks=[])
    )
    assert status == 413
    assert b"Request payload too large (max 64KB)" in body
    assert b"x-request-id" in resp_headers
    assert resp_headers[b"x-request-id"] == custom_req_id.encode("latin1")


def test_actual_body_greater_than_64kb_omitted_content_length_returns_413():
    """Actual body > 64 KiB with Content-Length omitted triggers 413."""
    custom_req_id = "omitted-cl-too-large-02"
    headers = [
        (b"content-type", b"application/json"),
        (b"x-request-id", custom_req_id.encode("latin1")),
    ]
    chunks = [{"type": "http.request", "body": b"A" * 70000, "more_body": False}]
    status, resp_headers, body = asyncio.run(
        _raw_asgi_post(app, headers=headers, chunks=chunks)
    )
    assert status == 413
    assert b"Request payload too large (max 64KB)" in body
    assert resp_headers[b"x-request-id"] == custom_req_id.encode("latin1")


def test_actual_body_greater_than_64kb_falsely_small_content_length_returns_413():
    """Actual body > 64 KiB while declared Content-Length is falsely smaller triggers 413."""
    custom_req_id = "false-cl-too-large-03"
    headers = [
        (b"content-length", b"50"),  # Deceptive small Content-Length
        (b"content-type", b"application/json"),
        (b"x-request-id", custom_req_id.encode("latin1")),
    ]
    chunks = [{"type": "http.request", "body": b"B" * 70000, "more_body": False}]
    status, resp_headers, body = asyncio.run(
        _raw_asgi_post(app, headers=headers, chunks=chunks)
    )
    assert status == 413
    assert b"Request payload too large (max 64KB)" in body
    assert resp_headers[b"x-request-id"] == custom_req_id.encode("latin1")


def test_body_delivered_over_multiple_asgi_chunks_exceeding_limit_returns_413():
    """Body delivered across multiple ASGI receive chunks whose total exceeds 64 KiB triggers 413."""
    custom_req_id = "multichunk-too-large-04"
    headers = [
        (b"content-type", b"application/json"),
        (b"x-request-id", custom_req_id.encode("latin1")),
    ]
    chunks = [
        {"type": "http.request", "body": b"C" * 40000, "more_body": True},
        {"type": "http.request", "body": b"D" * 30000, "more_body": False},
    ]
    status, resp_headers, body = asyncio.run(
        _raw_asgi_post(app, headers=headers, chunks=chunks)
    )
    assert status == 413
    assert b"Request payload too large (max 64KB)" in body
    assert resp_headers[b"x-request-id"] == custom_req_id.encode("latin1")


def test_body_exactly_at_or_below_boundary_passes_size_middleware(monkeypatch):
    """Body exactly at 65,536 bytes (64 KiB) passes size middleware without 413."""
    monkeypatch.setenv("HARSHU_AUTH_DISABLED", "true")
    # Build valid JSON up to exactly 65536 bytes
    prefix = b'{"question":"hi","padding":"'
    suffix = b'"}'
    padding_needed = 65536 - len(prefix) - len(suffix)
    exact_body = prefix + (b"p" * padding_needed) + suffix
    assert len(exact_body) == 65536

    monkeypatch.setattr(
        "harshu_ai_os.api.main.execute_request",
        lambda q: {"answer": "Processed successfully", "complexity": "simple", "workflow_used": "direct"},
    )

    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", b"65536"),
        (b"x-request-id", b"boundary-exact-64kb"),
    ]
    chunks = [{"type": "http.request", "body": exact_body, "more_body": False}]
    status, resp_headers, body = asyncio.run(
        _raw_asgi_post(app, headers=headers, chunks=chunks)
    )
    assert status == 200, f"Expected 200 at exact 64 KiB boundary, got {status}: {body}"
    assert b"Processed successfully" in body


def test_normal_json_request_reaches_endpoint(monkeypatch):
    """Normal JSON request below 64 KiB reaches endpoint and parses correctly."""
    monkeypatch.setenv("HARSHU_AUTH_DISABLED", "true")
    monkeypatch.setattr(
        "harshu_ai_os.api.main.execute_request",
        lambda q: {"answer": f"Echo: {q}", "complexity": "simple", "workflow_used": "direct"},
    )

    resp = client.post("/ask", json={"question": "What is Python?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Echo: What is Python?"


def test_oversized_body_contents_never_appear_in_logs():
    """Payload bytes from rejected requests must never leak into server logs."""
    from harshu_ai_os.tests.test_observability import LogCaptureHandler

    handler = LogCaptureHandler()
    obs_logger = logging.getLogger("harshu_ai_os")
    obs_logger.addHandler(handler)
    obs_logger.setLevel(logging.INFO)

    secret_oversized_content = "TOP_SECRET_OVERSIZED_PAYLOAD_CANARY_VALUE_XYZ"
    oversized_body = (secret_oversized_content * 1600).encode("utf-8")
    assert len(oversized_body) > 65536

    try:
        req_id = "oversized-no-log-canary"
        headers = [
            (b"content-type", b"application/json"),
            (b"x-request-id", req_id.encode("latin1")),
        ]
        chunks = [{"type": "http.request", "body": oversized_body, "more_body": False}]
        status, resp_headers, body = asyncio.run(
            _raw_asgi_post(app, headers=headers, chunks=chunks)
        )
        assert status == 413
        assert resp_headers[b"x-request-id"] == req_id.encode("latin1")

        all_logs = " ".join(handler.formatted_lines)
        assert secret_oversized_content not in all_logs
        assert "TOP_SECRET" not in all_logs
    finally:
        obs_logger.removeHandler(handler)
