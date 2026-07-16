from litellm import completion
from harshu_ai_os.llm.messages import build_messages


SYSTEM_PROMPT = (
    "You are the Harshu AI OS runtime assistant. "
    "Answer clearly and concisely. Unless the user asks for detail, "
    "keep the response under 150 words."
)


def call_llm(route: dict, user_prompt: str)-> str:

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

    response = completion(**completion_args)

    return response.choices[0].message.content