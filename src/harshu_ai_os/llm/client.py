from litellm import completion
from harshu_ai_os.llm.messages import build_messages
from harshu_ai_os.llm.exceptions import LLMServiceError
from litellm.exceptions import ServiceUnavailableError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(ServiceUnavailableError),
)
def make_llm_call(completion_args: dict):
    return completion(**completion_args)


SYSTEM_PROMPT = (
    "You are the Harshu AI OS runtime assistant. "
    "Answer clearly and concisely. Unless the user asks for detail, "
    "keep the response under 150 words."
)


def call_llm(route: dict, user_prompt: str) -> str:
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

        response = make_llm_call(completion_args)
        return response.choices[0].message.content

    except ServiceUnavailableError:
        raise LLMServiceError(
            "AI service is temporarily unavailable. Please try again."
        )
