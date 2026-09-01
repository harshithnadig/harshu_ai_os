"""Focused deterministic tests for Harshu AI OS Observability v1."""

import asyncio
import json
import logging
import uuid

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harshu_ai_os.api.main import app
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.observability.context import get_request_id
from harshu_ai_os.observability.logging import StructuredJsonFormatter
from harshu_ai_os.observability.middleware import ObservabilityMiddleware


class LogCaptureHandler(logging.Handler):
    """Custom logging handler to capture structured log events during tests."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.formatted_lines: list[str] = []
        self.formatter = StructuredJsonFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.formatted_lines.append(self.formatter.format(record))


@pytest.fixture
def log_capture():
    """Attach LogCaptureHandler to harshu_ai_os logger hierarchy."""
    handler = LogCaptureHandler()
    obs_logger = logging.getLogger("harshu_ai_os")
    obs_logger.addHandler(handler)
    obs_logger.setLevel(logging.INFO)
    yield handler
    obs_logger.removeHandler(handler)


client = TestClient(app)


# ======================================================================
# 1. Request without X-Request-ID gets generated UUID & response header
# ======================================================================

def test_request_without_x_request_id_gets_generated_uuid():
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers

    req_id = response.headers["x-request-id"]
    # Ensure it is a valid UUID
    parsed_uuid = uuid.UUID(req_id)
    assert str(parsed_uuid) == req_id


# ======================================================================
# 2. Valid caller-supplied X-Request-ID is propagated
# ======================================================================

def test_valid_caller_supplied_x_request_id_propagated():
    custom_id = "harshu-test-001"
    response = client.get("/health", headers={"X-Request-ID": custom_id})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == custom_id


# ======================================================================
# 3. Completion log contains same request ID as response and valid JSON
# ======================================================================

def test_completion_log_contains_same_request_id_and_fields(log_capture):
    custom_id = "req-correlation-999"
    response = client.get("/health", headers={"X-Request-ID": custom_id})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == custom_id

    # Find the http_request_completed log
    completed_logs = [
        json.loads(line)
        for line in log_capture.formatted_lines
        if "http_request_completed" in line
    ]
    assert len(completed_logs) >= 1
    log_entry = completed_logs[-1]

    assert log_entry["event"] == "http_request_completed"
    assert log_entry["request_id"] == custom_id
    assert log_entry["method"] == "GET"
    assert log_entry["path"] == "/health"
    assert log_entry["status_code"] == 200
    assert "timestamp" in log_entry
    assert "duration_ms" in log_entry
    assert log_entry["duration_ms"] >= 0.0


# ======================================================================
# 4. HTTP status codes logged correctly across scenarios
# ======================================================================

def test_http_status_codes_logged_correctly(log_capture):
    # 200 OK
    client.get("/health", headers={"X-Request-ID": "status-200"})
    # 400 Bad Request
    client.post("/ask", json={"question": "   "}, headers={"X-Request-ID": "status-400"})
    # 422 Unprocessable Entity
    client.post("/ask", json={}, headers={"X-Request-ID": "status-422"})

    logs_by_id = {
        entry["request_id"]: entry
        for line in log_capture.formatted_lines
        for entry in [json.loads(line)]
        if entry.get("event") == "http_request_completed"
    }

    assert logs_by_id["status-200"]["status_code"] == 200
    assert logs_by_id["status-400"]["status_code"] == 400
    assert logs_by_id["status-422"]["status_code"] == 422


# ======================================================================
# 5. duration_ms exists and is >= 0
# ======================================================================

def test_duration_ms_is_non_negative(log_capture):
    client.get("/health", headers={"X-Request-ID": "latency-check"})

    matching = [
        json.loads(line)
        for line in log_capture.formatted_lines
        if json.loads(line).get("request_id") == "latency-check"
        and json.loads(line).get("event") == "http_request_completed"
    ]
    assert len(matching) == 1
    entry = matching[0]
    assert isinstance(entry["duration_ms"], (float, int))
    assert entry["duration_ms"] >= 0.0


# ======================================================================
# 6. Separate requests receive independent IDs
# ======================================================================

def test_separate_requests_receive_independent_ids():
    ids = set()
    for _ in range(15):
        resp = client.get("/health")
        req_id = resp.headers["x-request-id"]
        assert req_id not in ids
        ids.add(req_id)
    assert len(ids) == 15


# ======================================================================
# 7. Concurrent requests do not leak request IDs
# ======================================================================

@pytest.mark.anyio
async def test_concurrent_requests_do_not_leak_request_ids():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        async def send_req(i: int):
            req_id = f"concurrent-test-req-{i:03d}"
            resp = await async_client.get("/health", headers={"X-Request-ID": req_id})
            return i, req_id, resp.headers.get("x-request-id")

        tasks = [send_req(i) for i in range(30)]
        results = await asyncio.gather(*tasks)

        for i, sent_id, returned_id in results:
            assert returned_id == sent_id, f"Request {i} expected {sent_id} but got {returned_id}"


# ======================================================================
# 8. Malformed / overly long supplied request IDs safely handled
# ======================================================================

def test_malformed_and_overly_long_request_ids_safely_handled():
    # Overly long ID (> 128 chars)
    long_id = "a" * 200
    resp = client.get("/health", headers={"X-Request-ID": long_id})
    assert resp.status_code == 200
    returned_id = resp.headers["x-request-id"]
    assert returned_id != long_id
    # Must be replaced with a valid UUID
    uuid.UUID(returned_id)

    # Injection / CRLF attempt
    crlf_id = "test-id\r\nX-Injected: evil"
    resp = client.get("/health", headers={"X-Request-ID": crlf_id})
    returned_id = resp.headers["x-request-id"]
    assert returned_id != crlf_id
    uuid.UUID(returned_id)

    # Invalid special characters
    special_id = "test<script>alert(1)</script>"
    resp = client.get("/health", headers={"X-Request-ID": special_id})
    returned_id = resp.headers["x-request-id"]
    assert returned_id != special_id
    uuid.UUID(returned_id)

    # Whitespace only
    ws_id = "    "
    resp = client.get("/health", headers={"X-Request-ID": ws_id})
    returned_id = resp.headers["x-request-id"]
    assert returned_id != ws_id
    uuid.UUID(returned_id)


# ======================================================================
# 9. Safe error logging on unhandled exceptions
# ======================================================================

def test_unhandled_exception_safe_error_logging(log_capture):
    # Build isolated mini test app with unhandled exception
    test_app = FastAPI()
    test_app.add_middleware(ObservabilityMiddleware)

    @test_app.get("/crash")
    def crash():
        raise RuntimeError("database_password_secret_12345")

    crash_client = TestClient(test_app, raise_server_exceptions=False)
    resp = crash_client.get("/crash", headers={"X-Request-ID": "crash-test-id"})

    assert resp.status_code == 500

    failed_logs = [
        json.loads(line)
        for line in log_capture.formatted_lines
        if json.loads(line).get("event") == "http_request_failed"
        and json.loads(line).get("request_id") == "crash-test-id"
    ]
    assert len(failed_logs) == 1
    err_entry = failed_logs[0]

    assert err_entry["event"] == "http_request_failed"
    assert err_entry["request_id"] == "crash-test-id"
    assert err_entry["method"] == "GET"
    assert err_entry["path"] == "/crash"
    assert err_entry["error_type"] == "RuntimeError"
    assert err_entry["duration_ms"] >= 0.0
    # Crucial privacy assertion: secret exception message must NOT be in the log line
    assert "database_password_secret_12345" not in log_capture.formatted_lines[-1]


# ======================================================================
# 10. Privacy/Security: sensitive request values not logged
# ======================================================================

def test_sensitive_request_values_not_present_in_logs(log_capture):
    secret_auth = "Bearer sk-test-secret-token-xyz"
    secret_cookie = "session=secret-cookie-12345"

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "privacy-test-id",
            "Authorization": secret_auth,
            "Cookie": secret_cookie,
        },
    )
    assert response.status_code == 200

    all_logs = " ".join(log_capture.formatted_lines)
    assert secret_auth not in all_logs
    assert secret_cookie not in all_logs
    assert "sk-test-secret-token-xyz" not in all_logs
    assert "secret-cookie-12345" not in all_logs


# ======================================================================
# 11. Endpoint compatibility & ContextVar access during request
# ======================================================================

def test_contextvar_accessible_in_route():
    test_app = FastAPI()
    test_app.add_middleware(ObservabilityMiddleware)

    @test_app.get("/inspect-context")
    def inspect():
        active_id = get_request_id()
        return {"context_request_id": active_id}

    tc = TestClient(test_app)
    resp = tc.get("/inspect-context", headers={"X-Request-ID": "context-prop-01"})
    assert resp.status_code == 200
    assert resp.json()["context_request_id"] == "context-prop-01"
    assert resp.headers["x-request-id"] == "context-prop-01"


# ======================================================================
# 12. Privacy/Security: handled LLMServiceError does not log raw message
# ======================================================================

@pytest.mark.parametrize(
    "endpoint,mock_target",
    [
        ("/ask", "harshu_ai_os.api.main.execute_request"),
        ("/ask/rag", "harshu_ai_os.api.main.answer_with_chroma_rag"),
        ("/ask/agent", "harshu_ai_os.api.main.run_agent_loop"),
    ],
)
def test_handled_llm_service_error_safe_structured_logging(
    log_capture, monkeypatch, endpoint, mock_target
):
    secret_message = "provider_secret_DO_NOT_LOG_93847"
    custom_request_id = f"sec-test-{endpoint.replace('/', '-')}"

    def mock_fail(*args, **kwargs):
        raise LLMServiceError(secret_message)

    monkeypatch.setattr(mock_target, mock_fail)
    if endpoint in ("/ask/rag", "/ask/agent"):
        monkeypatch.setattr(
            "harshu_ai_os.api.main.choose_request_route",
            lambda q: (
                type("Classification", (), {"complexity": "general"})(),
                {"model": "test-model"},
            ),
        )

    response = client.post(
        endpoint,
        json={"question": "What is the secret?"},
        headers={"X-Request-ID": custom_request_id},
    )

    # 1. Response status remains 503
    assert response.status_code == 503
    # 2. Response body remains the existing safe detail
    assert response.json() == {"detail": "AI service temporarily unavailable"}
    # 3. X-Request-ID is preserved
    assert response.headers["x-request-id"] == custom_request_id

    # 4. Structured log contains a safe llm_service_error event
    error_logs = [
        json.loads(line)
        for line in log_capture.formatted_lines
        if json.loads(line).get("event") == "llm_service_error"
        and json.loads(line).get("request_id") == custom_request_id
    ]
    assert len(error_logs) == 1
    err_entry = error_logs[0]

    # 5. Log contains error_type
    assert err_entry["error_type"] == "LLMServiceError"
    # 6. Log contains the same request ID
    assert err_entry["request_id"] == custom_request_id
    assert err_entry["level"] == "ERROR"

    # 7. Synthetic secret DOES NOT occur anywhere in captured logs
    all_logs_text = "\n".join(log_capture.formatted_lines)
    assert secret_message not in all_logs_text
