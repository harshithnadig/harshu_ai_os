"""Deterministic Test Suite for the Bounded ReAct Agent Loop.

Tests verify state transitions, tool execution permissions, error handling,
and budget termination without requiring live external model endpoints.
"""

from unittest.mock import MagicMock, patch
import pytest

from harshu_ai_os.agents.loop import (
    DEFAULT_MAX_STEPS,
    execute_single_tool,
    run_agent_loop,
)
from harshu_ai_os.llm.tools import WEB_SEARCH_TOOL_SCHEMA


# ==============================================================================
# 1. Helper Unit Tests (execute_single_tool boundary)
# ==============================================================================

def test_execute_single_tool_success():
    """Allowed tool runs properly and returns observation string and sources."""
    fake_tools = {
        "web_search": lambda query: {
            "content": f"Results for: {query}",
            "sources": [{"title": "Docs", "url": "https://example.com/docs"}],
        }
    }
    obs, sources = execute_single_tool(
        "web_search",
        '{"query": "Python 3.14"}',
        available_tools=fake_tools,
    )
    assert "Results for: Python 3.14" in obs
    assert len(sources) == 1
    assert sources[0]["url"] == "https://example.com/docs"


def test_execute_single_tool_unauthorized():
    """Unlisted tool is rejected safely with error observation."""
    fake_tools = {"allowed_tool": lambda: "ok"}
    obs, sources = execute_single_tool(
        "forbidden_rm_rf",
        '{"path": "/"}',
        available_tools=fake_tools,
    )
    assert "Error: Tool 'forbidden_rm_rf' is not allowed" in obs
    assert sources == []


def test_execute_single_tool_malformed_json():
    """Malformed JSON arguments are caught cleanly and return error observation."""
    fake_tools = {"web_search": lambda query: "ok"}
    obs, sources = execute_single_tool(
        "web_search",
        '{"query": "broken json...',
        available_tools=fake_tools,
    )
    assert "Malformed JSON arguments" in obs
    assert sources == []


# ==============================================================================
# 2. Scenarios for Harshu's Bounded Agent Loop
# ==============================================================================

def _make_mock_tool_call(call_id: str, func_name: str, args_json: str):
    """Helper to create a standard OpenAI tool call mock."""
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = func_name
    tool_call.function.arguments = args_json
    return tool_call


def _make_mock_response(content: str | None = None, tool_calls: list | None = None):
    """Helper to create a standard LiteLLM completion response mock."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    return MagicMock(choices=[MagicMock(message=message)])


@patch("harshu_ai_os.agents.loop.make_llm_call")
def test_agent_loop_immediate_final_answer(mock_make_call):
    """Scenario 1: Model immediately returns a final answer (0 tool calls)."""
    mock_make_call.return_value = _make_mock_response(
        content="FastAPI is a modern, fast Python web framework."
    )

    route = {"model": "openai/harshu-general", "max_tokens": 500}
    result = run_agent_loop(
        route=route,
        user_prompt="What is FastAPI in one sentence?",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools={"web_search": lambda query: {}},
        max_steps=5,
    )

    assert mock_make_call.call_count == 1
    assert "FastAPI" in result["answer"]
    assert result["steps_taken"] == 0
    assert result["tool_calls_count"] == 0
    assert result["tool_used"] is False
    assert result["stopped_reason"] in ["direct_answer", "completed"]


@patch("harshu_ai_os.agents.loop.make_llm_call")
def test_agent_loop_single_tool_and_answer(mock_make_call):
    """Scenario 2: Model requests one tool, receives observation, then answers."""
    # Round 1: Model requests web search
    call_1 = _make_mock_tool_call("call_web_1", "web_search", '{"query": "Python latest release"}')
    resp_1 = _make_mock_response(content=None, tool_calls=[call_1])

    # Round 2: Model receives observation and outputs final answer
    resp_2 = _make_mock_response(
        content="According to official search evidence, Python 3.14 is the latest release."
    )
    mock_make_call.side_effect = [resp_1, resp_2]

    fake_tools = {
        "web_search": lambda query: {
            "content": "[1] Python 3.14 Released: Official documentation.",
            "sources": [{"title": "Python Releases", "url": "https://python.org/3.14"}],
        }
    }

    route = {"model": "openai/harshu-tools", "max_tokens": 500}
    result = run_agent_loop(
        route=route,
        user_prompt="What is the latest release of Python?",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=fake_tools,
        max_steps=5,
    )

    assert mock_make_call.call_count == 2
    assert "Python 3.14" in result["answer"]
    assert result["steps_taken"] == 1
    assert result["tool_calls_count"] == 1
    assert result["tool_used"] is True
    assert len(result["tool_sources"]) == 1
    assert result["tool_sources"][0]["url"] == "https://python.org/3.14"
    assert result["stopped_reason"] == "completed"


@patch("harshu_ai_os.agents.loop.make_llm_call")
def test_agent_loop_max_steps_budget_exceeded(mock_make_call):
    """Scenario 3: Model keeps requesting tools until max_steps budget is reached."""
    max_steps_budget = 3

    # Rounds 1, 2, 3: Model requests continuous tool calls
    call_a = _make_mock_tool_call("c1", "search_docs", '{"q": "step 1"}')
    call_b = _make_mock_tool_call("c2", "search_docs", '{"q": "step 2"}')
    call_c = _make_mock_tool_call("c3", "search_docs", '{"q": "step 3"}')

    # Round 4 (Forced Synthesis): Model must answer without tools
    final_resp = _make_mock_response(
        content="Based on available multi-step evidence gathered so far, here is the synthesis."
    )

    mock_make_call.side_effect = [
        _make_mock_response(content=None, tool_calls=[call_a]),
        _make_mock_response(content=None, tool_calls=[call_b]),
        _make_mock_response(content=None, tool_calls=[call_c]),
        final_resp,
    ]

    fake_tools = {
        "search_docs": lambda q: {"content": f"Evidence for {q}", "sources": []}
    }

    route = {"model": "openai/harshu-tools", "max_tokens": 500}
    result = run_agent_loop(
        route=route,
        user_prompt="Investigate complex system behavior across multiple steps.",
        tools=[{"type": "function", "function": {"name": "search_docs"}}],
        available_tools=fake_tools,
        max_steps=max_steps_budget,
    )

    # Must make 3 tool iterations + 1 final forced synthesis call = 4 LLM calls
    assert mock_make_call.call_count == max_steps_budget + 1
    assert result["steps_taken"] == max_steps_budget
    assert result["tool_calls_count"] == 3
    assert result["stopped_reason"] == "max_steps_exceeded"
    assert "Based on available multi-step evidence" in result["answer"]

    # Verify that the final synthesis call had tools disabled
    final_call_args = mock_make_call.call_args_list[-1][0][0]
    assert "tools" not in final_call_args or final_call_args.get("tools") is None


@patch("harshu_ai_os.agents.loop.make_llm_call")
def test_agent_loop_unauthorized_tool_handled_safely(mock_make_call):
    """Scenario 4: Model requests an unlisted tool; loop provides error observation and continues."""
    bad_call = _make_mock_tool_call("call_hack", "delete_all_files", '{"target": "/"}')
    resp_1 = _make_mock_response(content=None, tool_calls=[bad_call])
    resp_2 = _make_mock_response(
        content="I cannot perform unauthorized filesystem deletions."
    )
    mock_make_call.side_effect = [resp_1, resp_2]

    # Only safe_read is allowed
    fake_tools = {"safe_read": lambda: "safe content"}

    route = {"model": "openai/harshu-tools", "max_tokens": 500}
    result = run_agent_loop(
        route=route,
        user_prompt="Please delete all server files.",
        tools=[{"type": "function", "function": {"name": "delete_all_files"}}],
        available_tools=fake_tools,
        max_steps=3,
    )

    assert mock_make_call.call_count == 2
    assert "cannot perform" in result["answer"]
    # Check that error observation was passed to round 2 LLM call
    second_call_args = mock_make_call.call_args_list[1][0][0]
    messages = second_call_args["messages"]
    tool_msg = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"][0]
    assert "Error: Tool 'delete_all_files' is not allowed" in tool_msg["content"]


@patch("harshu_ai_os.agents.loop.make_llm_call")
def test_agent_loop_malformed_tool_arguments_handled_safely(mock_make_call):
    """Scenario 5: Model produces malformed JSON arguments; loop feeds error back to model."""
    bad_arg_call = _make_mock_tool_call("call_broken", "web_search", '{"query": "unclosed string...')
    resp_1 = _make_mock_response(content=None, tool_calls=[bad_arg_call])
    resp_2 = _make_mock_response(
        content="I adjusted my query and answered the prompt."
    )
    mock_make_call.side_effect = [resp_1, resp_2]

    fake_tools = {"web_search": lambda query: "results"}

    route = {"model": "openai/harshu-tools", "max_tokens": 500}
    result = run_agent_loop(
        route=route,
        user_prompt="Search something for me.",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=fake_tools,
        max_steps=3,
    )

    assert mock_make_call.call_count == 2
    assert "answered the prompt" in result["answer"]
    second_call_args = mock_make_call.call_args_list[1][0][0]
    messages = second_call_args["messages"]
    tool_msg = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"][0]
    assert "Malformed JSON arguments" in tool_msg["content"]
