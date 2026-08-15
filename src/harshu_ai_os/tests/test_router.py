import pytest
from harshu_ai_os.llm.router import (
    choose_route,
    SIMPLE_MODEL,
    GENERAL_MODEL,
    REASONING_MODEL,
)


def test_choose_route_invalid():
    with pytest.raises(ValueError):
        choose_route("unknown")


def test_choose_route_simple():
    route = choose_route("simple")

    assert route["model"] == SIMPLE_MODEL
    assert route["max_tokens"] == 150


def test_choose_route_general():
    route = choose_route("general")

    assert route["model"] == GENERAL_MODEL
    assert route["max_tokens"] == 500


def test_choose_route_complex():
    route = choose_route("complex")

    assert route["model"] == REASONING_MODEL
    assert route["max_tokens"] == 2000


def test_classify_task_with_model_mocked(monkeypatch):
    """Verify classify_task_with_model parses structured JSON from gateway."""
    from unittest.mock import MagicMock
    from harshu_ai_os.core import get_omniroute_config
    from harshu_ai_os.llm.router import classify_task_with_model, CLASSIFIER_MODEL

    base_url, _ = get_omniroute_config()

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"complexity": "simple", "needs_current_information": false, "needs_tool": false}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    captured_kwargs = {}

    def fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    monkeypatch.setattr("harshu_ai_os.llm.router.completion", fake_completion)

    result = classify_task_with_model("What is 2+2?")

    assert result.complexity == "simple"
    assert result.needs_current_information is False
    assert result.needs_tool is False
    assert captured_kwargs["model"] == CLASSIFIER_MODEL
    assert captured_kwargs["api_base"] == base_url


