"""Model-provider boundary for direct answers and LangChain RAG workflows.

The router owns *which* logical route to choose. This module turns that route
into one provider call or one LangChain chat-model object.
"""

import litellm
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from litellm import completion
from litellm.exceptions import ServiceUnavailableError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harshu_ai_os.llm.exceptions import LLMServiceError

# Routes intentionally contain provider-specific controls. LiteLLM drops only
# controls unsupported by the selected provider instead of rejecting the call.
litellm.drop_params = True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(ServiceUnavailableError),
    reraise=True,
)
def make_llm_call(completion_args: dict):
    """Retry only transient provider-unavailable failures for direct calls."""
    return completion(**completion_args)


SYSTEM_PROMPT = (
    "You are the Harshu AI OS runtime assistant. "
    "Answer clearly and concisely. Unless the user asks for detail, "
    "keep the response under 150 words."
)

LANGCHAIN_PROVIDER_NAMES = {
    "gemini": "google_genai",
}


def build_messages(
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    """Keep direct LiteLLM calls on one consistent system/user message shape."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(route: dict, user_prompt: str) -> str:
    """Send a direct answer request through LiteLLM using one chosen route."""
    try:
        messages = build_messages(
            SYSTEM_PROMPT,
            user_prompt,
        )
        completion_args = {
            "model": route["model"],
            "messages": messages,
            "max_completion_tokens": route["max_tokens"],
            "timeout": 30,
            "temperature": 0.0,
        }
        if "reasoning_effort" in route:
            completion_args["reasoning_effort"] = route["reasoning_effort"]
        if "thinking" in route:
            completion_args["thinking"] = route["thinking"]

        response = make_llm_call(completion_args)
        return response.choices[0].message.content

    except ServiceUnavailableError:
        raise LLMServiceError(
            "AI service is temporarily unavailable. Please try again."
        )


def to_langchain_identifier(route: dict) -> str:
    """Translate the existing LiteLLM route without changing router ownership."""
    provider, model_name = route["model"].split("/", maxsplit=1)
    provider = LANGCHAIN_PROVIDER_NAMES.get(provider, provider)
    return f"{provider}:{model_name}"


def create_chat_model_from_route(route: dict) -> BaseChatModel:
    """Create the LangChain model used by composed workflows such as RAG."""
    model_options = {
        "temperature": 0,
        "max_tokens": route["max_tokens"],
        "timeout": 30,
        "max_retries": 3,
    }

    # LangChain integrations use provider-native names for reasoning controls.
    if "reasoning_effort" in route:
        model_options["reasoning_effort"] = route["reasoning_effort"]

    thinking = route.get("thinking")
    if thinking and "budget_tokens" in thinking:
        model_options["thinking_budget"] = thinking["budget_tokens"]

    return init_chat_model(
        to_langchain_identifier(route),
        **model_options,
    )
