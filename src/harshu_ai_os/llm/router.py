"""Classify each request and select a logical model route."""

from dotenv import load_dotenv
from litellm import completion
from typing import Literal

from pydantic import BaseModel

from harshu_ai_os.llm.client import build_messages


# The classifier uses a direct LiteLLM call because it is intentionally small
# and does not need RAG prompt composition.
load_dotenv()

SIMPLE_MODEL = "groq/llama-3.1-8b-instant"
GENERAL_MODEL = "gemini/gemini-2.5-flash"
REASONING_MODEL = "groq/openai/gpt-oss-20b"
CLASSIFIER_MODEL = "gemini/gemini-2.5-flash"


class TaskClassification(BaseModel):
    """Structured routing decision returned by the classifier model."""

    complexity: Literal["simple", "general", "complex"]
    needs_current_information: bool
    needs_tool: bool


def classify_task_with_model(user_prompt: str) -> TaskClassification:
    """Classify a request before any model route is selected."""
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
        'Example: "Say hello." -> '
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

    return TaskClassification.model_validate_json(raw_result)


def choose_route(task_type: str) -> dict:
    """Map a logical complexity level to provider and generation controls."""
    if task_type == "simple":
        return {
            "model": SIMPLE_MODEL,
            "max_tokens": 80,
        }

    if task_type == "general":
        return {
            "model": GENERAL_MODEL,
            "max_tokens": 500,
            "thinking": {
                "type": "disabled",
                "budget_tokens": 0,
            },
        }

    if task_type == "complex":
        return {
            "model": REASONING_MODEL,
            "max_tokens": 1000,
            "reasoning_effort": "medium",
        }

    raise ValueError(f"Unknown task type: {task_type}")
