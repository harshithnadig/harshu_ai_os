"""Classify each request and select a logical model route."""

import re
from typing import Literal

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel

from harshu_ai_os.core import get_omniroute_config
from harshu_ai_os.llm.client import build_messages

load_dotenv()

# OmniRoute logical roles
SIMPLE_MODEL = "openai/harshu-general"
GENERAL_MODEL = "openai/harshu-general"
REASONING_MODEL = "openai/harshu-reasoning"
CLASSIFIER_MODEL = "openai/harshu-classifier"
TOOLS_MODEL = "openai/harshu-tools"
JUDGE_MODEL = "openai/harshu-judge"


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

    base_url, api_key = get_omniroute_config()
    response = completion(
        model=CLASSIFIER_MODEL,
        api_base=base_url,
        api_key=api_key,
        messages=messages,
        max_tokens=1000,
        temperature=0.0,
        timeout=30,
    )

    raw_result = (response.choices[0].message.content or "").strip()
    json_match = re.search(r"\{.*\}", raw_result, re.DOTALL)
    if json_match:
        raw_result = json_match.group(0)

    return TaskClassification.model_validate_json(raw_result)


def choose_route(task_type: str) -> dict:
    """Map a logical complexity level to OmniRoute logical role and token controls."""
    if task_type == "simple":
        return {
            "model": SIMPLE_MODEL,
            "max_tokens": 150,
        }

    if task_type == "general":
        return {
            "model": GENERAL_MODEL,
            "max_tokens": 500,
        }

    if task_type == "complex":
        return {
            "model": REASONING_MODEL,
            "max_tokens": 2000,
        }

    raise ValueError(f"Unknown task type: {task_type}")

