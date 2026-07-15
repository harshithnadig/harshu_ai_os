import litellm
from dotenv import load_dotenv
from litellm import completion
import sys
from harshu_ai_os.llm.messages import build_messages
import json

load_dotenv()
litellm.drop_params = True

SYSTEM_PROMPT = (
    "You are the Harshu AI OS runtime assistant. "
    "Answer clearly and concisely. Unless the user asks for detail, "
    "keep the response under 150 words."
)

# Temporary learning model.
# Groq free/developer shutdown: 2026-08-16.
# MODEL_NAME = "groq/llama-3.1-8b-instant"
SIMPLE_MODEL = "groq/llama-3.1-8b-instant"
GENERAL_MODEL = "gemini/gemini-2.5-flash"
REASONING_MODEL = "groq/openai/gpt-oss-20b"
CLASSIFIER_MODEL = "gemini/gemini-2.5-flash-lite"



def classify_task_with_model(user_prompt: str) -> dict:
    classifier_system_prompt = (
    "You classify user requests for an AI router. "
    "Do not answer the request. "
    "Return only valid JSON with exactly these keys: "
    '"complexity", "needs_current_information", and "needs_tool". '

    'Use "simple" only for greetings, short factual transformations, '
    "very short extraction, or one-line direct answers. "

    'Use "general" for explanations, summaries, ordinary coding help, '
    "examples, and normal multi-paragraph answers. "

    'Use "complex" for architecture, security design, advanced debugging, '
    "multi-step planning, trade-off analysis, or difficult reasoning. "

    '"complexity" must be exactly "simple", "general", or "complex". '
    '"needs_current_information" and "needs_tool" must be true or false. '
    "Do not use Markdown fences. Do not add any other keys."
    )

    classifier_system_prompt += (
    ' Example: "Say hello." -> '
    '{"complexity":"simple","needs_current_information":false,"needs_tool":false}. '

    'Example: "Explain Python dictionaries with one example." -> '
    '{"complexity":"general","needs_current_information":false,"needs_tool":false}. '

    'Example: "Design a secure RAG architecture." -> '
    '{"complexity":"complex","needs_current_information":false,"needs_tool":false}.'
    )
    messages = build_messages(
        classifier_system_prompt,
        user_prompt,
    )

    response = completion(
        model=CLASSIFIER_MODEL,
        messages=messages,
        max_completion_tokens=100,
        reasoning_effort="none",
        temperature=0.0,
        timeout=30,
    )

    raw_result = response.choices[0].message.content
    classification = json.loads(raw_result)
    print(type(classification))
    print(classification)
    print(classification["complexity"])
    return classification

def choose_route(task_type: str) -> dict:
    if task_type == "simple":
        return {
            "model": SIMPLE_MODEL,
            "max_tokens": 80,
           
        }
    if task_type == "general":
        return {
            "model": GENERAL_MODEL,
            "max_tokens": 500,
        
        }
    if task_type == "complex":
        return {
            "model": REASONING_MODEL,
            "max_tokens": 1000,
            "reasoning_effort": "medium",
        }


def classify_task(user_prompt: str) -> str:
    prompt = user_prompt.lower()

    complexity_signals = [
        "design",
        "explain",
        "secure",
        "architecture",
        "trade-off",
        "production",
        "debug",
        "optimize",
        "threat model",
        "rag",
        "database",
    ]

    signal_count = 0

    for signal in complexity_signals:
        if signal in prompt:
            signal_count += 1

    if signal_count >= 2:
        return "complex"

    if signal_count == 1 or len(user_prompt) > 50:
        return "general"
    return "simple"


def call_model(route: dict, user_prompt: str):
    messages = build_messages(SYSTEM_PROMPT, user_prompt)

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
    return response

if __name__ == "__main__":
    user_prompt = input("Ask me anything: ")
    classification = classify_task_with_model(user_prompt)
    print("Classification:", classification)
# if __name__ == "__main__":
#     sys.stdout.reconfigure(encoding="utf-8")
#     user_prompt = input("Ask me anything: ")
#     task_type = classify_task_with_model(user_prompt)

#     route = choose_route(task_type)

#     response = call_model(route, user_prompt)

#     print("Task type:", task_type)
#     print("Chosen model:", route["model"])
#     print("Token budget:", route["max_tokens"])
#     print("Reply:", response.choices[0].message.content)
#     print("Finish reason:", response.choices[0].finish_reason)
#     print("Usage:", response.usage)
