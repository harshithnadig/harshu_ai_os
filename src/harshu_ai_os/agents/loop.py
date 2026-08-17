"""Bounded ReAct-style Agent Loop Scaffolding for Harshu AI OS.

Implements the iterative DECIDE -> ACT -> OBSERVE -> DECIDE ... -> STOP pattern
with an explicit tool-step budget constraint.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from harshu_ai_os.core import get_omniroute_config
from harshu_ai_os.llm.client import (
    SYSTEM_PROMPT,
    TOOL_SYNTHESIS_SYSTEM_PROMPT,
    build_messages,
    make_llm_call,
)

# Standard agent execution constraints
DEFAULT_MAX_STEPS: int = 3

AGENT_SYSTEM_PROMPT: str = (
    "You are an autonomous ReAct agent in Harshu AI OS.\n"
    "Solve the user's task using step-by-step tool decisions:\n"
    "1. DECIDE: If you need information or actions, call an available tool.\n"
    "   - 'rag_lookup': Search Harshu AI OS internal/local indexed project knowledge, notes, or architecture.\n"
    "   - 'web_search': Search the live web for current facts, release statuses, news, or external verification.\n"
    "   - If an observation does not contain the required facts or you need current external information, switch to the other appropriate tool rather than repeatedly querying the same source.\n"
    "2. OBSERVE: You will receive the tool result as an observation.\n"
    "3. SYNTHESIZE: When you have sufficient evidence to answer, return your final answer directly without calling further tools.\n"
    "4. GROUNDING: Base all factual answers strictly on verified observations. If evidence is insufficient, say so rather than inventing."
)


@dataclass
class AgentStep:
    """Records one completed tool action and observation in the agent loop."""

    step_index: int
    tool_name: str
    tool_args: dict[str, Any]
    observation: str
    sources: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Structured outcome of a bounded agent loop execution."""

    answer: str
    steps_taken: int
    tool_calls_count: int
    tool_sources: list[dict[str, str]] = field(default_factory=list)
    stopped_reason: str = (
        "completed"  # "direct_answer" | "completed" | "max_steps_exceeded"
    )
    steps: list[AgentStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard Harshu AI OS API dictionary format."""
        return {
            "answer": self.answer,
            "steps_taken": self.steps_taken,
            "tool_calls_count": self.tool_calls_count,
            "tool_sources": self.tool_sources,
            "stopped_reason": self.stopped_reason,
            "tool_used": self.tool_calls_count > 0,
        }


def execute_single_tool(
    func_name: str,
    args_raw: Any,
    available_tools: Optional[Dict[str, Callable]] = None,
) -> tuple[str, list[dict[str, str]]]:
    """Safely validate, parse arguments, and execute a tool against the allowlist.

    Helper boundary for Harshu's loop.

    Returns:
        tuple of (observation_string, list_of_collected_sources)
    """
    if not available_tools or func_name not in available_tools:
        return (
            f"Error: Tool '{func_name}' is not allowed or not found in available tools.",
            [],
        )

    # 1. Parse Arguments safely
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError as exc:
            return f"Error: Malformed JSON arguments for tool '{func_name}': {exc}", []
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}

    # 2. Execute Python Tool
    tool_func = available_tools[func_name]
    try:
        tool_raw = tool_func(**args)
    except Exception as exc:
        return f"Error executing tool '{func_name}': {exc}", []

    # 3. Extract output and sources
    collected_sources: list[dict[str, str]] = []
    if isinstance(tool_raw, dict):
        output_text = str(tool_raw.get("content", tool_raw))
        raw_sources = tool_raw.get("sources", [])
        if isinstance(raw_sources, list):
            collected_sources.extend(raw_sources)
    else:
        output_text = str(tool_raw)

    return output_text, collected_sources


def run_agent_loop(
    route: dict[str, Any],
    user_prompt: str,
    tools: Optional[list[dict[str, Any]]] = None,
    available_tools: Optional[dict[str, Callable]] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: Optional[str] = None,
    required_tools: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Execute a bounded ReAct-style agent loop (Decide -> Act -> Observe -> Decide ... -> Stop).

    Args:
        route: Routing dictionary (e.g. {"model": "openai/harshu-tools", "max_tokens": 600}).
        user_prompt: The user query or instruction.
        tools: List of OpenAI-compatible tool JSON schemas available to the model.
        available_tools: Dict mapping allowed tool names to executable Python functions.
        max_steps: Maximum allowable tool-execution rounds before forced synthesis.
        system_prompt: Optional custom system prompt override.
        required_tools: Optional set of tool names that must be executed before final synthesis.

    Returns:
        dict matching the AgentResult schema:
            - "answer": str (the final response generated by the model)
            - "steps_taken": int (number of tool-execution rounds completed)
            - "tool_calls_count": int (total number of tools invoked)
            - "tool_sources": list[dict] (collected source citations)
            - "stopped_reason": str ("direct_answer" | "completed" | "max_steps_exceeded")
            - "tool_used": bool (True if at least one tool was invoked)
    """
    messages = build_messages(system_prompt or AGENT_SYSTEM_PROMPT, user_prompt)
    base_url, api_key = get_omniroute_config()
    steps_taken = 0
    tool_calls_count = 0
    collected_sources = []
    step_history = []
    executed_tool_names: set[str] = set()
    required_tool_set = set(required_tools) if required_tools else set()

    while steps_taken < max_steps:
        missing_required = required_tool_set - executed_tool_names
        model_name = "openai/harshu-tools" if tools else route["model"]

        # If exactly one required tool remains missing, constrain tool_choice to it
        if len(missing_required) == 1:
            missing_tool_name = next(iter(missing_required))
            tool_choice = {
                "type": "function",
                "function": {"name": missing_tool_name},
            }
        else:
            tool_choice = "auto"

        completion_args = {
            "model": model_name,
            "api_base": base_url,
            "api_key": api_key,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_completion_tokens": route.get("max_tokens", 500),
        }
        response = make_llm_call(completion_args)
        message = response.choices[0].message
        tool_calls = message.tool_calls

        # Handle text output without tool calls
        if message.content and not tool_calls:
            # If required tools are still missing, do not accept early completion
            if missing_required:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Required tool coverage is incomplete. Please gather evidence using: "
                            f"{', '.join(missing_required)}."
                        ),
                    }
                )
                steps_taken += 1
                continue

            return AgentResult(
                answer=message.content,
                steps_taken=steps_taken,
                tool_calls_count=tool_calls_count,
                tool_sources=collected_sources,
                stopped_reason=(
                    "direct_answer" if tool_calls_count == 0 else "completed"
                ),
                steps=step_history,
            ).to_dict()

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
        )
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                args_raw = tool_call.function.arguments
                observation, sources = execute_single_tool(
                    function_name, args_raw, available_tools
                )
                if not observation.startswith("Error: Tool '"):
                    executed_tool_names.add(function_name)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": observation,
                    }
                )
                step_history.append(
                    AgentStep(
                        step_index=steps_taken,
                        tool_name=function_name,
                        tool_args=args_raw,
                        observation=observation,
                        sources=sources,
                    )
                )
                collected_sources.extend(sources)
                tool_calls_count += 1
        steps_taken += 1

        if steps_taken >= max_steps:
            messages[0]["content"] = TOOL_SYNTHESIS_SYSTEM_PROMPT
            completion_args = {
                "model": route["model"],
                "api_base": base_url,
                "api_key": api_key,
                "messages": messages,
                "max_completion_tokens": max(route.get("max_tokens", 500), 1000),
            }
            response = make_llm_call(completion_args)
            message = response.choices[0].message
            return AgentResult(
                answer=message.content,
                steps_taken=steps_taken,
                tool_calls_count=tool_calls_count,
                tool_sources=collected_sources,
                stopped_reason="max_steps_exceeded",
                steps=step_history,
            ).to_dict()

