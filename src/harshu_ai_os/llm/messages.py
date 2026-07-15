from litellm import completion
from litellm.exceptions import ServiceUnavailableError
import time
from dotenv import load_dotenv
import os

load_dotenv()

model = "gemini-2.5-flash-lite"
model_name = f"gemini/{model}"
system_prompt = "You are the Harshu AI OS runtime assistant."
user_prompt = (
    "Reply in one short sentence: two modules are active and one module is inactive."
)


def generate_reply(
    model_name: str,
    messages: list[dict[str, str]],
) -> str:
    try:
        response = completion(
            model=model_name, messages=messages, max_tokens=50, timeout=30
        )
        return response.choices[0].message.content
    except ServiceUnavailableError:
        return "The AI provider is temporarily unavailable. Please try again."


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:

    system_message = {"role": "system", "content": system_prompt}

    user_message = {"role": "user", "content": user_prompt}

    return [system_message, user_message]


if __name__ == "__main__":
    messages = build_messages(system_prompt, user_prompt)

    print(messages)
    print(model_name)

    start_time = time.time()

    reply = generate_reply(model_name, messages)

    end_time = time.time()
    total_time = end_time - start_time

    print(reply)
    print(f"Total time: {total_time:.2f} seconds")
