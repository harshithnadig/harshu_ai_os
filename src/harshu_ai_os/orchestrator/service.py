"""Unified request orchestrator for Harshu AI OS.

Coordinates planning and deterministic execution between:
1. DIRECT: Standard LLM generation without tools or grounding.
2. AGENT: Multi-step ReAct agent using web_search and/or rag_lookup.
3. STRICT_RAG: Grounded retrieval with distance filter, sufficiency judge, and citations.
"""

import json
import re
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel

from harshu_ai_os.agents.loop import run_agent_loop
from harshu_ai_os.core import get_logger, get_omniroute_config
from harshu_ai_os.llm.client import build_messages, call_llm
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.llm.router import CLASSIFIER_MODEL, choose_route
from harshu_ai_os.llm.tools import (
    AVAILABLE_TOOLS,
    RAG_LOOKUP_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
)
from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.service import (
    DEFAULT_MAXIMUM_DISTANCE,
    answer_with_chroma_rag,
)

load_dotenv()
logger = get_logger(__name__)


class RequestPlan(BaseModel):
    """Structured plan defining request complexity, source, and grounding intent."""

    complexity: Literal["simple", "general", "complex"] = "general"
    information_source: Literal["none", "web", "internal", "mixed"] = "none"
    strict_internal_grounding: bool = False
    reason: Optional[str] = None


def plan_request(question: str) -> RequestPlan:
    """Classify user query and create a structured RequestPlan."""
    planner_system_prompt = (
        "You are the query planner for Harshu AI OS. "
        "Analyze the user request and determine the execution plan. "
        "Do not answer the request. "
        "Return ONLY valid JSON with exactly these keys: "
        '"complexity", "information_source", "strict_internal_grounding", and "reason".\n\n'
        "Definitions:\n"
        '1. "complexity": "simple" | "general" | "complex"\n'
        '   - "simple": greetings, short definitions, basic math, simple facts, one-line transforms.\n'
        '   - "general": standard explanations, coding help, summaries, normal reasoning.\n'
        '   - "complex": intricate architecture, deep debugging, multi-step planning, high reasoning.\n'
        '2. "information_source": "none" | "web" | "internal" | "mixed"\n'
        '   - "none": standard coding, concepts, general knowledge, greetings that do not require external or project-specific lookups.\n'
        '   - "web": questions needing live/current external information (e.g. current release versions, recent news, real-time web info).\n'
        '   - "internal": questions asking about Harshu AI OS internal architecture, codebase, notes, or project components.\n'
        '   - "mixed": questions requiring BOTH internal Harshu AI OS knowledge AND external web searches (e.g. project notes plus current external status/versions).\n'
        '3. "strict_internal_grounding": true | false\n'
        "   - true ONLY when the user explicitly requests strict grounding restricted only to indexed project documents, requiring citation and abstention "
        '(e.g., "Strictly based only on the indexed project documents...", "Use only project documents and cite evidence...").\n'
        '   - MUST be false if the query requires external or current information ("mixed" or "web"), or mentions "according to project notes" without explicit strict restriction.\n'
        '4. "reason": brief string explaining the choice.\n\n'
        "Do not use Markdown fences. Return raw JSON only."
    )

    messages = build_messages(planner_system_prompt, question)
    base_url, api_key = get_omniroute_config()

    try:
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

        return RequestPlan.model_validate_json(raw_result)
    except Exception as error:
        logger.error("Failed to generate a valid request plan: %s", error)
        if isinstance(error, LLMServiceError):
            raise
        raise LLMServiceError(f"Failed to generate execution plan: {error}") from error


def choose_workflow(plan: RequestPlan) -> str:
    """Deterministically map RequestPlan to one of: strict_rag, agent, direct."""
    if plan.strict_internal_grounding:
        return "strict_rag"
    if plan.information_source in ("web", "internal", "mixed"):
        return "agent"
    return "direct"


def execute_request(
    question: str,
    plan: Optional[RequestPlan] = None,
    collection: Any = None,
    embedding_client: Any = None,
) -> dict:
    """Execute the planned workflow and return unified response metadata."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if plan is None:
        plan = plan_request(question)

    workflow = choose_workflow(plan)
    route = choose_route(plan.complexity)

    logger.info(
        "orchestrator_plan: complexity=%s info_source=%s strict_rag=%s -> workflow=%s model=%s",
        plan.complexity,
        plan.information_source,
        plan.strict_internal_grounding,
        workflow,
        route["model"],
    )

    if workflow == "strict_rag":
        if collection is None:
            collection = get_notes_collection()
        if embedding_client is None:
            embedding_client = get_embedding_client()

        rag_result = answer_with_chroma_rag(
            collection=collection,
            client=embedding_client,
            question=question,
            route=route,
            maximum_distance=DEFAULT_MAXIMUM_DISTANCE,
        )

        return {
            "answer": rag_result.get("answer", ""),
            "complexity": plan.complexity,
            "workflow_used": "strict_rag",
            "model": route["model"],
            "tool_used": False,
            "tool_calls_count": 0,
            "tool_sources": [],
            "citations": rag_result.get("citations", []),
            "abstained": rag_result.get("abstained", False),
            "abstention_reason": rag_result.get("abstention_reason"),
            "judge_reason": rag_result.get("judge_reason"),
            "stopped_reason": "rag_grounded" if not rag_result.get("abstained") else "rag_abstained",
            "steps_taken": 0,
            "plan": plan,
        }

    if workflow == "agent":
        system_prompt = None
        required_tools = None
        if plan.information_source == "mixed":
            required_tools = {"rag_lookup", "web_search"}
            system_prompt = (
                "You are an autonomous ReAct agent in Harshu AI OS.\n"
                "PLAN CONTEXT: This request requires BOTH internal project knowledge and external/current information.\n"
                "1. Gather evidence for both parts before final synthesis:\n"
                "   - Use 'rag_lookup' to find Harshu AI OS internal/project knowledge and notes.\n"
                "   - Use 'web_search' to verify current real-time status, releases, or external facts.\n"
                "2. If an observation from one tool does not answer the remaining requirements, switch to the other tool rather than repeating the same query.\n"
                "3. Ground your final synthesis in the verified facts from both observations."
            )

        agent_result = run_agent_loop(
            route=route,
            user_prompt=question,
            tools=[WEB_SEARCH_TOOL_SCHEMA, RAG_LOOKUP_TOOL_SCHEMA],
            available_tools=AVAILABLE_TOOLS,
            system_prompt=system_prompt,
            required_tools=required_tools,
        )

        return {
            "answer": agent_result.get("answer", ""),
            "complexity": plan.complexity,
            "workflow_used": "agent",
            "model": route["model"],
            "tool_used": agent_result.get("tool_used", False),
            "tool_calls_count": agent_result.get("tool_calls_count", 0),
            "tool_sources": agent_result.get("tool_sources", []),
            "citations": [],
            "abstained": False,
            "abstention_reason": None,
            "judge_reason": None,
            "stopped_reason": agent_result.get("stopped_reason", "completed"),
            "steps_taken": agent_result.get("steps_taken", 0),
            "plan": plan,
        }

    # Default: direct workflow
    direct_result = call_llm(
        route=route,
        user_prompt=question,
    )
    answer_text = (
        direct_result.get("answer", "")
        if isinstance(direct_result, dict)
        else str(direct_result)
    )

    return {
        "answer": answer_text,
        "complexity": plan.complexity,
        "workflow_used": "direct",
        "model": route["model"],
        "tool_used": False,
        "tool_calls_count": 0,
        "tool_sources": [],
        "citations": [],
        "abstained": False,
        "abstention_reason": None,
        "judge_reason": None,
        "stopped_reason": "direct_answer",
        "steps_taken": 0,
        "plan": plan,
    }
