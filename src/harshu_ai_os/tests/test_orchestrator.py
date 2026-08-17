"""Tests for Harshu AI OS Request Orchestrator."""

from unittest.mock import MagicMock
import pytest

from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.orchestrator.service import (
    RequestPlan,
    choose_workflow,
    execute_request,
    plan_request,
)


def test_choose_workflow_deterministic():
    """Verify python deterministic workflow selection rules."""
    # 1. strict_internal_grounding == True -> strict_rag
    plan_rag = RequestPlan(
        complexity="general",
        information_source="internal",
        strict_internal_grounding=True,
    )
    assert choose_workflow(plan_rag) == "strict_rag"

    # Even if information_source is web or none, strict_internal_grounding wins
    plan_rag_override = RequestPlan(
        complexity="complex",
        information_source="web",
        strict_internal_grounding=True,
    )
    assert choose_workflow(plan_rag_override) == "strict_rag"

    # 2. information_source in ("web", "internal", "mixed") -> agent
    plan_web = RequestPlan(
        complexity="simple",
        information_source="web",
        strict_internal_grounding=False,
    )
    assert choose_workflow(plan_web) == "agent"

    plan_internal = RequestPlan(
        complexity="general",
        information_source="internal",
        strict_internal_grounding=False,
    )
    assert choose_workflow(plan_internal) == "agent"

    plan_mixed = RequestPlan(
        complexity="complex",
        information_source="mixed",
        strict_internal_grounding=False,
    )
    assert choose_workflow(plan_mixed) == "agent"

    # 3. otherwise -> direct
    plan_direct = RequestPlan(
        complexity="general",
        information_source="none",
        strict_internal_grounding=False,
    )
    assert choose_workflow(plan_direct) == "direct"


def test_plan_request_success(monkeypatch):
    """Test plan_request parsing valid json from classifier."""
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = '{"complexity":"complex","information_source":"mixed","strict_internal_grounding":false,"reason":"Needs both"}'
    fake_response.choices = [fake_choice]

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.completion",
        lambda **kwargs: fake_response,
    )

    plan = plan_request("Compare local Chroma DB notes with latest Python 3.14 release.")
    assert plan.complexity == "complex"
    assert plan.information_source == "mixed"
    assert plan.strict_internal_grounding is False
    assert plan.reason == "Needs both"


def test_plan_request_cases(monkeypatch):
    """Test planner classifications for internal, strict RAG, mixed, and web questions."""
    cases = [
        (
            '{"complexity":"general","information_source":"internal","strict_internal_grounding":false,"reason":"Internal project notes"}',
            "According to Harshu AI OS project notes, what query router categories are used?",
            "internal",
            False,
        ),
        (
            '{"complexity":"general","information_source":"internal","strict_internal_grounding":true,"reason":"Explicit strict restriction to indexed documents"}',
            "Strictly based only on the indexed project documents, what API framework is used?",
            "internal",
            True,
        ),
        (
            '{"complexity":"complex","information_source":"mixed","strict_internal_grounding":false,"reason":"Requires project notes and live web verification"}',
            "According to Harshu AI OS project notes, which model role handles reasoning, and is the underlying model still current today?",
            "mixed",
            False,
        ),
        (
            '{"complexity":"simple","information_source":"web","strict_internal_grounding":false,"reason":"Live current release info"}',
            "What is the latest stable Python release today?",
            "web",
            False,
        ),
    ]

    for mock_json, question, expected_source, expected_strict in cases:
        fake_response = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = mock_json
        fake_response.choices = [fake_choice]

        monkeypatch.setattr(
            "harshu_ai_os.orchestrator.service.completion",
            lambda **kwargs: fake_response,
        )

        plan = plan_request(question)
        assert plan.information_source == expected_source
        assert plan.strict_internal_grounding == expected_strict

        if expected_source == "mixed":
            assert choose_workflow(plan) == "agent"


def test_plan_request_fails_closed_on_malformed_json(monkeypatch):
    """Test plan_request fails closed (raises LLMServiceError) when json parsing fails."""
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "Unparseable non-json response text"
    fake_response.choices = [fake_choice]

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.completion",
        lambda **kwargs: fake_response,
    )

    with pytest.raises(LLMServiceError, match="Failed to generate execution plan"):
        plan_request("What is the latest release?")


def test_planner_failure_does_not_invoke_direct_llm(monkeypatch):
    """Prove malformed planner JSON does NOT invoke direct LLM generation."""
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "Invalid JSON"
    fake_response.choices = [fake_choice]

    direct_llm_called = False

    def fake_call_llm(*args, **kwargs):
        nonlocal direct_llm_called
        direct_llm_called = True
        return "Unwanted direct answer"

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.completion",
        lambda **kwargs: fake_response,
    )
    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.call_llm",
        fake_call_llm,
    )

    with pytest.raises(LLMServiceError):
        execute_request("Some question")

    assert direct_llm_called is False


def test_planner_failure_does_not_invoke_agent(monkeypatch):
    """Prove malformed planner JSON does NOT invoke the agent."""
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "Invalid JSON"
    fake_response.choices = [fake_choice]

    agent_called = False

    def fake_agent(*args, **kwargs):
        nonlocal agent_called
        agent_called = True
        return {}

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.completion",
        lambda **kwargs: fake_response,
    )
    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.run_agent_loop",
        fake_agent,
    )

    with pytest.raises(LLMServiceError):
        execute_request("Some question")

    assert agent_called is False


def test_planner_failure_does_not_invoke_strict_rag(monkeypatch):
    """Prove malformed planner JSON does NOT invoke strict RAG."""
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "Invalid JSON"
    fake_response.choices = [fake_choice]

    rag_called = False

    def fake_rag(*args, **kwargs):
        nonlocal rag_called
        rag_called = True
        return {}

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.completion",
        lambda **kwargs: fake_response,
    )
    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.answer_with_chroma_rag",
        fake_rag,
    )

    with pytest.raises(LLMServiceError):
        execute_request("Some question")

    assert rag_called is False


def test_execute_request_direct(monkeypatch):
    """Test execute_request with direct workflow."""
    plan = RequestPlan(
        complexity="simple",
        information_source="none",
        strict_internal_grounding=False,
    )

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.call_llm",
        lambda route, user_prompt: "Paris is the capital of France.",
    )

    result = execute_request("What is the capital of France?", plan=plan)

    assert result["workflow_used"] == "direct"
    assert result["answer"] == "Paris is the capital of France."
    assert result["complexity"] == "simple"
    assert result["model"] == "openai/harshu-general"
    assert result["tool_used"] is False
    assert result["tool_calls_count"] == 0
    assert result["citations"] == []
    assert result["abstained"] is False


def test_execute_request_mixed_passes_plan_context(monkeypatch):
    """Test execute_request with mixed workflow passes plan-aware system prompt and required_tools."""
    plan = RequestPlan(
        complexity="complex",
        information_source="mixed",
        strict_internal_grounding=False,
    )

    captured_system_prompt = None
    captured_required_tools = None

    def fake_agent_loop(route, user_prompt, tools, available_tools, system_prompt=None, required_tools=None):
        nonlocal captured_system_prompt, captured_required_tools
        captured_system_prompt = system_prompt
        captured_required_tools = required_tools
        return {
            "answer": "Combined answer.",
            "steps_taken": 2,
            "tool_calls_count": 2,
            "tool_sources": [],
            "stopped_reason": "completed",
            "tool_used": True,
        }

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.run_agent_loop",
        fake_agent_loop,
    )

    result = execute_request("Compare local notes with latest Python version", plan=plan)

    assert result["workflow_used"] == "agent"
    assert captured_required_tools == {"rag_lookup", "web_search"}
    assert captured_system_prompt is not None
    assert "PLAN CONTEXT" in captured_system_prompt
    assert "rag_lookup" in captured_system_prompt
    assert "web_search" in captured_system_prompt


def test_execute_request_web_only_agent(monkeypatch):
    """Test execute_request with web-only workflow does not set required_tools."""
    plan = RequestPlan(
        complexity="simple",
        information_source="web",
        strict_internal_grounding=False,
    )

    captured_required_tools = "INITIAL"

    def fake_agent_loop(route, user_prompt, tools, available_tools, system_prompt=None, required_tools=None):
        nonlocal captured_required_tools
        captured_required_tools = required_tools
        return {
            "answer": "Web answer.",
            "steps_taken": 1,
            "tool_calls_count": 1,
            "tool_sources": [],
            "stopped_reason": "completed",
            "tool_used": True,
        }

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.run_agent_loop",
        fake_agent_loop,
    )

    result = execute_request("What is latest Python version?", plan=plan)

    assert result["workflow_used"] == "agent"
    assert captured_required_tools is None


def test_execute_request_internal_only_agent(monkeypatch):
    """Test execute_request with internal-only workflow does not set required_tools."""
    plan = RequestPlan(
        complexity="general",
        information_source="internal",
        strict_internal_grounding=False,
    )

    captured_required_tools = "INITIAL"

    def fake_agent_loop(route, user_prompt, tools, available_tools, system_prompt=None, required_tools=None):
        nonlocal captured_required_tools
        captured_required_tools = required_tools
        return {
            "answer": "Internal answer.",
            "steps_taken": 1,
            "tool_calls_count": 1,
            "tool_sources": [],
            "stopped_reason": "completed",
            "tool_used": True,
        }

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.run_agent_loop",
        fake_agent_loop,
    )

    result = execute_request("What are router categories in project notes?", plan=plan)

    assert result["workflow_used"] == "agent"
    assert captured_required_tools is None


def test_execute_request_strict_rag(monkeypatch):
    """Test execute_request with strict RAG workflow."""
    plan = RequestPlan(
        complexity="general",
        information_source="internal",
        strict_internal_grounding=True,
    )

    fake_collection = MagicMock()
    fake_embedding_client = MagicMock()

    def fake_rag(collection, client, question, route, maximum_distance):
        return {
            "answer": "Harshu AI OS uses FastAPI.",
            "abstained": False,
            "abstention_reason": None,
            "judge_reason": "Direct match in documents.",
            "context": "FastAPI is used.",
            "citations": [
                {
                    "source": "overview.txt",
                    "chunk_id": "chunk_0",
                    "chunk_index": 0,
                    "distance": 0.15,
                }
            ],
        }

    monkeypatch.setattr(
        "harshu_ai_os.orchestrator.service.answer_with_chroma_rag",
        fake_rag,
    )

    result = execute_request(
        "According to project documents, what does Harshu AI OS use?",
        plan=plan,
        collection=fake_collection,
        embedding_client=fake_embedding_client,
    )

    assert result["workflow_used"] == "strict_rag"
    assert result["answer"] == "Harshu AI OS uses FastAPI."
    assert result["abstained"] is False
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk_0"
    assert result["stopped_reason"] == "rag_grounded"


def test_execute_request_empty_question():
    """Test that empty question raises ValueError."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        execute_request("   ")
