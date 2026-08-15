from langchain_core.language_models.fake_chat_models import FakeListChatModel

from harshu_ai_os.llm.client import (
    create_chat_model_from_route,
)
from harshu_ai_os.rag.service import (
    create_grounded_chat_prompt,
    create_grounded_text_chain,
)


def test_chat_model_factory_preserves_route_controls() -> None:
    route = {
        "model": "openai/harshu-general",
        "max_tokens": 500,
    }

    model = create_chat_model_from_route(route)

    assert model is not None
    assert model.model_name == "harshu-general"
    assert model.max_tokens == 500
    assert "http" in model.base_url


def test_chat_model_factory_handles_logical_roles() -> None:
    model = create_chat_model_from_route(
        {
            "model": "openai/harshu-reasoning",
            "max_tokens": 2000,
        }
    )

    assert model.model_name == "harshu-reasoning"
    assert model.max_tokens == 2000



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
