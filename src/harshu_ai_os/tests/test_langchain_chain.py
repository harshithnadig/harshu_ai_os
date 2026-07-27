from langchain_core.language_models.fake_chat_models import FakeListChatModel

from harshu_ai_os.llm.client import (
    create_chat_model_from_route,
    to_langchain_identifier,
)
from harshu_ai_os.rag.service import (
    create_grounded_chat_prompt,
    create_grounded_text_chain,
)


def test_langchain_identifier_preserves_provider_model_paths() -> None:
    route = {
        "model": "groq/openai/gpt-oss-20b",
        "max_tokens": 1000,
    }

    assert to_langchain_identifier(route) == "groq:openai/gpt-oss-20b"


def test_chat_model_factory_preserves_route_controls(monkeypatch) -> None:
    captured = {}

    def fake_init_chat_model(identifier, **options):
        captured["identifier"] = identifier
        captured["options"] = options
        return object()

    monkeypatch.setattr(
        "harshu_ai_os.llm.client.init_chat_model",
        fake_init_chat_model,
    )

    route = {
        "model": "gemini/gemini-2.5-flash",
        "max_tokens": 500,
        "thinking": {
            "type": "disabled",
            "budget_tokens": 0,
        },
    }

    model = create_chat_model_from_route(route)

    assert model is not None
    assert captured == {
        "identifier": "google_genai:gemini-2.5-flash",
        "options": {
            "temperature": 0,
            "max_tokens": 500,
            "timeout": 30,
            "max_retries": 3,
            "thinking_budget": 0,
        },
    }


def test_chat_model_factory_preserves_reasoning_effort(monkeypatch) -> None:
    captured = {}

    def fake_init_chat_model(identifier, **options):
        captured["identifier"] = identifier
        captured["options"] = options
        return object()

    monkeypatch.setattr(
        "harshu_ai_os.llm.client.init_chat_model",
        fake_init_chat_model,
    )

    create_chat_model_from_route(
        {
            "model": "groq/openai/gpt-oss-20b",
            "max_tokens": 1000,
            "reasoning_effort": "medium",
        }
    )

    assert captured["identifier"] == "groq:openai/gpt-oss-20b"
    assert captured["options"]["reasoning_effort"] == "medium"
    assert captured["options"]["max_retries"] == 3


def test_grounded_text_chain_returns_text() -> None:
    expected_answer = "ChromaDB stores and retrieves document embeddings."

    fake_model = FakeListChatModel(
        responses=[expected_answer],
    )

    chain = create_grounded_text_chain(fake_model)

    result = chain.invoke(
        {
            "context": expected_answer,
            "question": "What does ChromaDB do?",
        }
    )

    assert isinstance(result, str)
    assert result == expected_answer


def test_grounded_prompt_keeps_rules_separate_from_user_data() -> None:
    prompt = create_grounded_chat_prompt()

    formatted = prompt.invoke(
        {
            "context": "ChromaDB stores document embeddings.",
            "question": "What does ChromaDB store?",
        }
    )

    assert prompt.input_variables == ["context", "question"]
    assert "only the supplied context" in formatted.messages[0].content
    assert "ChromaDB stores document embeddings." in formatted.messages[1].content
