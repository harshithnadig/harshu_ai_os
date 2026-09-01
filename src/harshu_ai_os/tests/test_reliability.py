"""Tests for LLM Reliability Hardening: bounded retries, permanent vs transient failures, and fallback."""

from unittest.mock import MagicMock, patch
import pytest
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from harshu_ai_os.llm.client import call_llm, make_llm_call
from harshu_ai_os.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from harshu_ai_os.llm.router import choose_route


def test_transient_failure_retries_and_succeeds(monkeypatch):
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ServiceUnavailableError("Transient 503", model="test-model", llm_provider="openai")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Recovered successfully", tool_calls=None))]
        return mock_resp

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {"model": "test-model", "max_tokens": 100}
    answer = call_llm(route, "Test query")

    assert answer == "Recovered successfully"
    assert len(calls) == 2  # Proves exactly 1 retry succeeded


def test_transient_failure_exhaustion_raises_service_error(monkeypatch):
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs)
        raise ServiceUnavailableError("Permanent 503", model="test-model", llm_provider="openai")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {"model": "test-model", "max_tokens": 100}
    with pytest.raises(LLMServiceError, match="AI service is temporarily unavailable"):
        call_llm(route, "Test query")

    assert len(calls) == 3  # Bounded to 3 attempts


def test_permanent_auth_failure_does_not_retry(monkeypatch):
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs)
        raise AuthenticationError("Bad API key", model="test-model", llm_provider="openai")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {"model": "test-model", "max_tokens": 100}
    with pytest.raises(LLMAuthenticationError, match="AI service authentication failed"):
        call_llm(route, "Test query")

    assert len(calls) == 1  # Crucial: NO RETRIES on permanent auth failure


def test_timeout_translates_to_timeout_error(monkeypatch):
    def mock_completion(**kwargs):
        raise Timeout("Request timed out", model="test-model", llm_provider="openai")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {"model": "test-model", "max_tokens": 100}
    with pytest.raises(LLMTimeoutError, match="AI service request timed out"):
        call_llm(route, "Test query")


def test_rate_limit_translates_to_rate_limit_error(monkeypatch):
    def mock_completion(**kwargs):
        raise RateLimitError("Rate limit hit", model="test-model", llm_provider="openai")

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {"model": "test-model", "max_tokens": 100}
    with pytest.raises(LLMRateLimitError, match="AI service rate limit reached"):
        call_llm(route, "Test query")


def test_fallback_route_activation_on_primary_failure(monkeypatch):
    attempted_models = []

    def mock_completion(**kwargs):
        model = kwargs.get("model")
        attempted_models.append(model)
        if model == "primary-failing-model":
            raise ServiceUnavailableError("Primary down", model="primary-failing-model", llm_provider="openai")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Answer from fallback", tool_calls=None))]
        return mock_resp

    monkeypatch.setattr("harshu_ai_os.llm.client.completion", mock_completion)

    route = {
        "model": "primary-failing-model",
        "fallback_model": "backup-working-model",
        "max_tokens": 100,
    }
    answer = call_llm(route, "Test query")

    assert answer == "Answer from fallback"
    assert "primary-failing-model" in attempted_models
    assert "backup-working-model" in attempted_models
