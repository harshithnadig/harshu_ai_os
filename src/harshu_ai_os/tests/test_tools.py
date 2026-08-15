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
    assert "[1] Python 3.14 Released: New features in Python 3.14." in result["content"]
    assert "[2] FastAPI Updates: FastAPI latest release notes." in result["content"]
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
