"""Tests for LLM Reliability Hardening: bounded retries, permanent vs transient failures, and fallback."""

from unittest.mock import MagicMock
import httpx
import pytest
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMServiceError,
)


def _mock_success_response(text: str = "Success response"):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=text, tool_calls=None))]
    return mock_resp


def test_1_primary_service_unavailable_retries_then_activates_fallback(monkeypatch):
    """Primary 503 retries up to bound, then successfully activates configured fallback."""
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs.get("model"))
        if kwargs.get("model") == "primary-model":
            raise ServiceUnavailableError("Primary 503", model="primary-model", llm_provider="openai")
        return _mock_success_response("Answer from fallback")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-model",
        "fallback_model": "backup-model",
        "max_tokens": 100,
    }
    answer = call_llm(route, "Test query")

    assert answer == "Answer from fallback"
    # Primary model made exactly 3 bounded attempts
    assert calls.count("primary-model") == 3
    # Fallback model made exactly 1 attempt
    assert calls.count("backup-model") == 1
    assert len(calls) == 4


def test_2_authentication_error_fails_immediately_no_retry_no_fallback(monkeypatch):
    """Permanent AuthenticationError: exactly 1 call, NO retry, NO fallback."""
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs.get("model"))
        raise AuthenticationError("Bad API key", model="primary-model", llm_provider="openai")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-model",
        "fallback_model": "backup-model",
        "max_tokens": 100,
    }
    with pytest.raises(LLMAuthenticationError, match="AI service authentication failed"):
        call_llm(route, "Test query")

    assert len(calls) == 1
    assert calls == ["primary-model"]


def test_3_permission_denied_fails_immediately_no_retry_no_fallback(monkeypatch):
    """Permanent PermissionDeniedError: exactly 1 call, NO retry, NO fallback."""
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs.get("model"))
        dummy_resp = httpx.Response(403, request=httpx.Request("POST", "http://test"))
        raise PermissionDeniedError("Forbidden", "openai", "primary-model", response=dummy_resp)

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-model",
        "fallback_model": "backup-model",
        "max_tokens": 100,
    }
    with pytest.raises(LLMPermissionError, match="AI service access denied") as exc_info:
        call_llm(route, "Test query")

    assert isinstance(exc_info.value, LLMServiceError)
    assert len(calls) == 1
    assert calls == ["primary-model"]


def test_4_bad_request_fails_immediately_no_retry_no_fallback(monkeypatch):
    """Permanent BadRequestError: exactly 1 call, NO retry, NO fallback."""
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs.get("model"))
        dummy_resp = httpx.Response(400, request=httpx.Request("POST", "http://test"))
        raise BadRequestError("Bad request parameters", "primary-model", "openai", response=dummy_resp)

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-model",
        "fallback_model": "backup-model",
        "max_tokens": 100,
    }
    with pytest.raises(LLMBadRequestError, match="AI service rejected invalid request parameters") as exc_info:
        call_llm(route, "Test query")

    assert isinstance(exc_info.value, LLMServiceError)
    assert len(calls) == 1
    assert calls == ["primary-model"]


def test_5_timeout_retries_and_activates_fallback(monkeypatch):
    """Transient Timeout: bounded retries on primary, then fallback activates."""
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs.get("model"))
        if kwargs.get("model") == "primary-model":
            raise Timeout("Request timed out", model="primary-model", llm_provider="openai")
        return _mock_success_response("Answer after timeout fallback")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-model",
        "fallback_model": "backup-model",
        "max_tokens": 100,
    }
    answer = call_llm(route, "Test query")

    assert answer == "Answer after timeout fallback"
    assert calls.count("primary-model") == 3
    assert calls.count("backup-model") == 1


def test_6_unexpected_runtime_error_no_fallback(monkeypatch):
    """Unexpected non-transient exception: exactly 1 call, NO fallback, normalized LLMServiceError."""
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs.get("model"))
        raise RuntimeError("Internal unexpected error")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-model",
        "fallback_model": "backup-model",
        "max_tokens": 100,
    }
    with pytest.raises(LLMServiceError, match="AI service is temporarily unavailable"):
        call_llm(route, "Test query")

    assert len(calls) == 1
    assert calls == ["primary-model"]


def test_7_transient_recovery_on_retry_does_not_invoke_fallback(monkeypatch):
    """Primary recovers on retry 2: fallback model must NEVER be invoked."""
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs.get("model"))
        if len(calls) == 1:
            raise ServiceUnavailableError("Transient 503", model="primary-model", llm_provider="openai")
        return _mock_success_response("Recovered on retry 2")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-model",
        "fallback_model": "backup-model",
        "max_tokens": 100,
    }
    answer = call_llm(route, "Test query")

    assert answer == "Recovered on retry 2"
    assert len(calls) == 2
    assert all(m == "primary-model" for m in calls)
    assert "backup-model" not in calls


def test_rate_limit_translates_to_rate_limit_error(monkeypatch):
    """RateLimitError translates to LLMRateLimitError when no fallback configured."""
    def mock_completion(**kwargs):
        raise RateLimitError("Rate limit hit", model="test-model", llm_provider="openai")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {"model": "test-model", "max_tokens": 100}
    with pytest.raises(LLMRateLimitError, match="AI service rate limit reached"):
        call_llm(route, "Test query")
