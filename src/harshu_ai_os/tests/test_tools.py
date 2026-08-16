"""Tests for tool calling and web search tool execution."""

from unittest.mock import MagicMock, patch

from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.llm.tools import AVAILABLE_TOOLS, WEB_SEARCH_TOOL_SCHEMA, web_search


def test_web_search_empty_query():
    """Empty search queries return a friendly fallback structure."""
    res1 = web_search("")
    assert res1["content"] == "No search query provided."
    assert res1["sources"] == []

    res2 = web_search("   ")
    assert res2["content"] == "No search query provided."
    assert res2["sources"] == []


@patch("harshu_ai_os.llm.tools.DDGS")
def test_web_search_success(mock_ddgs_cls):
    """Successful search results are formatted into snippets and structured sources."""
    mock_instance = MagicMock()
    mock_instance.text.return_value = [
        {
            "title": "Python 3.14 Released",
            "body": "New features in Python 3.14.",
            "href": "https://python.org/news/3.14",
        },
        {
            "title": "FastAPI Updates",
            "body": "FastAPI latest release notes.",
            "href": "https://fastapi.tiangolo.com",
        },
    ]
    mock_ddgs_cls.return_value = mock_instance

    result = web_search("Python 3.14")
    assert "[1] Python 3.14 Released (Source: https://python.org/news/3.14): New features in Python 3.14." in result["content"]
    assert "[2] FastAPI Updates (Source: https://fastapi.tiangolo.com): FastAPI latest release notes." in result["content"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["title"] == "Python 3.14 Released"
    assert result["sources"][0]["url"] == "https://python.org/news/3.14"
    assert result["sources"][1]["title"] == "FastAPI Updates"
    assert result["sources"][1]["url"] == "https://fastapi.tiangolo.com"
    assert result["query"] == "Python 3.14"


@patch("harshu_ai_os.llm.tools.DDGS")
def test_web_search_no_results(mock_ddgs_cls):
    """Empty results list returns a clear fallback message and empty sources."""
    mock_instance = MagicMock()
    mock_instance.text.return_value = []
    mock_ddgs_cls.return_value = mock_instance

    result = web_search("Unknown query")
    assert result["content"] == "No web results found."
    assert result["sources"] == []


@patch("harshu_ai_os.llm.tools.DDGS")
def test_web_search_exception_handled(mock_ddgs_cls):
    """Network or parsing errors do not crash the application."""
    mock_instance = MagicMock()
    mock_instance.text.side_effect = Exception("Rate limit reached")
    mock_ddgs_cls.return_value = mock_instance

    result = web_search("Python")
    assert "Web search failed: Rate limit reached" in result["content"]
    assert result["sources"] == []


@patch("harshu_ai_os.llm.client.make_llm_call")
def test_call_llm_direct_answer_without_tools(mock_make_call):
    """Model answers directly when no tool call is requested."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Direct answer.", tool_calls=None))
    ]
    mock_make_call.return_value = mock_response

    route = {"model": "test-model", "max_tokens": 100}
    answer = call_llm(
        route,
        "What is 2+2?",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=AVAILABLE_TOOLS,
    )

    assert answer == "Direct answer."
    assert mock_make_call.call_count == 1


@patch("harshu_ai_os.llm.client.make_llm_call")
def test_call_llm_direct_answer_with_tool_info(mock_make_call):
    """Model answers directly and returns structured info when return_tool_info=True."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Direct answer.", tool_calls=None))
    ]
    mock_make_call.return_value = mock_response

    route = {"model": "test-model", "max_tokens": 100}
    result = call_llm(
        route,
        "What is 2+2?",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=AVAILABLE_TOOLS,
        return_tool_info=True,
    )

    assert result["answer"] == "Direct answer."
    assert result["tool_used"] is False
    assert result["tool_name"] is None
    assert result["tool_query"] is None
    assert result["tool_sources"] == []


@patch("harshu_ai_os.llm.client.make_llm_call")
def test_call_llm_with_web_search_dispatch(mock_make_call):
    """Model requests web_search, Python executes it, and model gives final answer."""
    # Step 1 response: Model requests web_search
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_abc123"
    mock_tool_call.function.name = "web_search"
    mock_tool_call.function.arguments = '{"query": "Latest AI news"}'

    first_message = MagicMock(content=None, tool_calls=[mock_tool_call])
    first_response = MagicMock(choices=[MagicMock(message=first_message)])

    # Step 2 response: Model produces final answer with the tool output
    second_message = MagicMock(content="Here is the latest AI news summary.")
    second_response = MagicMock(choices=[MagicMock(message=second_message)])

    mock_make_call.side_effect = [first_response, second_response]

    fake_tools = {
        "web_search": MagicMock(
            return_value={
                "content": "[1] AI News: GPT updates.",
                "sources": [{"title": "AI News", "url": "https://ai.example.com"}],
                "query": "Latest AI news",
            }
        ),
    }

    route = {"model": "test-model", "max_tokens": 100}
    result = call_llm(
        route,
        "What is the latest AI news?",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=fake_tools,
        return_tool_info=True,
    )

    assert result["answer"] == "Here is the latest AI news summary."
    assert result["tool_used"] is True
    assert result["tool_name"] == "web_search"
    assert result["tool_query"] == "Latest AI news"
    assert result["tool_sources"] == [{"title": "AI News", "url": "https://ai.example.com"}]
    assert mock_make_call.call_count == 2
    fake_tools["web_search"].assert_called_once_with(query="Latest AI news")


@patch("harshu_ai_os.llm.client.make_llm_call")
def test_call_llm_unauthorized_tool_handled_safely(mock_make_call):
    """Calls to unsupported or unauthorized tools return a safe error message."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_eval123"
    mock_tool_call.function.name = "unauthorized_tool"
    mock_tool_call.function.arguments = "{}"

    first_message = MagicMock(content=None, tool_calls=[mock_tool_call])
    first_response = MagicMock(choices=[MagicMock(message=first_message)])

    second_message = MagicMock(content="I could not run that tool.")
    second_response = MagicMock(choices=[MagicMock(message=second_message)])

    mock_make_call.side_effect = [first_response, second_response]

    route = {"model": "test-model", "max_tokens": 100}
    answer = call_llm(
        route,
        "Run evil code",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=AVAILABLE_TOOLS,
    )

    assert answer == "I could not run that tool."
    assert mock_make_call.call_count == 2


@patch("harshu_ai_os.llm.client.make_llm_call")
def test_call_llm_grounds_answer_on_web_search_evidence(mock_make_call):
    """Tool-augmented answers pass observations back to model to ensure grounded synthesis."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_py123"
    mock_tool_call.function.name = "web_search"
    mock_tool_call.function.arguments = '{"query": "Python latest release"}'

    first_message = MagicMock(content=None, tool_calls=[mock_tool_call])
    first_response = MagicMock(choices=[MagicMock(message=first_message)])

    second_message = MagicMock(content="According to official search results, Python 3.14.6 is the latest release.")
    second_response = MagicMock(choices=[MagicMock(message=second_message)])

    mock_make_call.side_effect = [first_response, second_response]

    fake_tools = {
        "web_search": MagicMock(
            return_value={
                "content": "[1] Python.org: Python 3.14.6 is released and available.",
                "sources": [{"title": "Python Source Releases", "url": "https://www.python.org/getit/source/"}],
                "query": "Python latest release",
            }
        ),
    }

    route = {"model": "test-model", "max_tokens": 100}
    result = call_llm(
        route,
        "What is the latest release of Python?",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=fake_tools,
        return_tool_info=True,
    )

    assert "Python 3.14.6" in result["answer"]
    assert result["tool_used"] is True
    assert result["tool_sources"][0]["url"] == "https://www.python.org/getit/source/"
    # Verify the tool output was included in messages sent to the second LLM call
    second_call_args = mock_make_call.call_args_list[1][0][0]
    messages_sent = second_call_args["messages"]
    tool_message = [m for m in messages_sent if isinstance(m, dict) and m.get("role") == "tool"][0]
    assert "Python 3.14.6 is released" in tool_message["content"]


@patch("harshu_ai_os.llm.client.make_llm_call")
def test_call_llm_enforces_primary_source_and_no_invention_contract_on_conflicts(mock_make_call):
    """When sources conflict, second pass prompt enforces primary-source preference and no-invention contract."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_conflict_99"
    mock_tool_call.function.name = "web_search"
    mock_tool_call.function.arguments = '{"query": "Python latest release"}'

    first_message = MagicMock(content=None, tool_calls=[mock_tool_call])
    first_response = MagicMock(choices=[MagicMock(message=first_message)])

    second_message = MagicMock(content="According to the official python.org release page, Python 3.14.6 is the latest stable release.")
    second_response = MagicMock(choices=[MagicMock(message=second_message)])

    mock_make_call.side_effect = [first_response, second_response]

    # Conflicting observations: official python.org says 3.14.6, random blog says 3.14.7
    conflicting_tool_content = (
        "[1] Python Source Releases (Source: https://www.python.org/getit/source/): Python 3.14.6 is the latest stable release.\n\n"
        "[2] Unofficial Tech Blog (Source: https://randomblog.example.com/python): The latest version is 3.14.7."
    )
    fake_tools = {
        "web_search": MagicMock(
            return_value={
                "content": conflicting_tool_content,
                "sources": [
                    {"title": "Python Source Releases", "url": "https://www.python.org/getit/source/"},
                    {"title": "Unofficial Tech Blog", "url": "https://randomblog.example.com/python"},
                ],
                "query": "Python latest release",
            }
        ),
    }

    route = {"model": "test-model", "max_tokens": 150}
    result = call_llm(
        route,
        "What is the latest stable release of Python?",
        tools=[WEB_SEARCH_TOOL_SCHEMA],
        available_tools=fake_tools,
        return_tool_info=True,
    )

    # Verify second-pass call structure and contract
    second_call_args = mock_make_call.call_args_list[1][0][0]
    messages = second_call_args["messages"]

    # 1. System prompt in second pass must be the rigorous synthesis contract
    system_prompt = messages[0]["content"]
    assert "CRITICAL GROUNDING RULES FOR SYNTHESIS" in system_prompt
    assert "Do NOT invent, extrapolate, or hallucinate" in system_prompt
    assert "prioritize authoritative primary/official sources" in system_prompt
    assert "state the discrepancy" in system_prompt

    # 2. Tool message contains the conflicting observation data
    tool_msg = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"][0]
    assert "https://www.python.org/getit/source/" in tool_msg["content"]
    assert "3.14.6" in tool_msg["content"]
    assert "3.14.7" in tool_msg["content"]

    # 3. Tools parameters are cleanly removed to prevent recursive tool loops
    assert "tools" not in second_call_args
    assert "tool_choice" not in second_call_args

    # 4. Result preserves tool activity and response
    assert result["tool_used"] is True
    assert len(result["tool_sources"]) == 2
    assert "3.14.6" in result["answer"]


