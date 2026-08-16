"""Model-provider boundary for direct answers and LangChain RAG workflows.

The router owns *which* logical route to choose. This module turns that route
into one provider call or one LangChain chat-model object via OmniRoute gateway.
"""

import json
import re
from typing import Any, List, Optional
import litellm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from litellm import completion
from litellm.exceptions import ServiceUnavailableError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harshu_ai_os.core import get_omniroute_config
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
    "Answer clearly, accurately, and concisely. Unless the user asks for detail, "
    "keep the response under 150 words."
)

TOOL_SYNTHESIS_SYSTEM_PROMPT = (
    "You are the Harshu AI OS runtime assistant.\n"
    "CRITICAL GROUNDING RULES FOR SYNTHESIS:\n"
    "1. Base your answer strictly on facts present in the provided tool observations.\n"
    "2. Do NOT invent, extrapolate, or hallucinate version numbers, dates, statistics, or names absent from those observations.\n"
    "3. When sources conflict, prioritize authoritative primary/official sources (e.g., official project domains or official documentation) over third-party blogs or aggregators.\n"
    "4. If reliable returned sources genuinely conflict and cannot be resolved, explicitly state the discrepancy rather than choosing an unsupported value.\n"
    "5. Keep the answer concise, accurate, and under 150 words."
)


def build_messages(
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    """Keep direct LiteLLM calls on one consistent system/user message shape."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(
    route: dict,
    user_prompt: str,
    tools: list[dict] | None = None,
    available_tools: dict | None = None,
    return_tool_info: bool = False,
) -> str | dict:
    """Send a direct answer request through OmniRoute gateway, with optional tool execution."""
    try:
        messages = build_messages(
            SYSTEM_PROMPT,
            user_prompt,
        )
        base_url, api_key = get_omniroute_config()
        model_name = "openai/harshu-tools" if tools else route["model"]

        completion_args = {
            "model": model_name,
            "api_base": base_url,
            "api_key": api_key,
            "messages": messages,
            "max_completion_tokens": route.get("max_tokens", 500),
            "timeout": 30,
            "temperature": 0.0,
        }
        if tools:
            completion_args["tools"] = tools
            completion_args["tool_choice"] = "auto"

        response = make_llm_call(completion_args)
        message = response.choices[0].message

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and isinstance(message, dict):
            tool_calls = message.get("tool_calls")

        if tool_calls and available_tools:
            messages.append(message)
            executed_name = None
            executed_query = None
            collected_sources = []

            for tool_call in tool_calls:
                func_name = (
                    tool_call.function.name
                    if hasattr(tool_call, "function")
                    else tool_call["function"]["name"]
                )
                args_raw = (
                    tool_call.function.arguments
                    if hasattr(tool_call, "function")
                    else tool_call["function"]["arguments"]
                )
                call_id = (
                    tool_call.id
                    if hasattr(tool_call, "id")
                    else tool_call.get("id", "call_1")
                )

                args = (
                    json.loads(args_raw)
                    if isinstance(args_raw, str)
                    else (args_raw or {})
                )

                if executed_name is None:
                    executed_name = func_name
                    executed_query = args.get("query")

                tool_func = available_tools.get(func_name)
                if tool_func:
                    tool_raw = tool_func(**args)
                    if isinstance(tool_raw, dict):
                        tool_output = tool_raw.get("content", str(tool_raw))
                        sources = tool_raw.get("sources", [])
                        if isinstance(sources, list):
                            collected_sources.extend(sources)
                    else:
                        tool_output = str(tool_raw)
                else:
                    tool_output = f"Error: Tool '{func_name}' is not allowed."

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": str(tool_output),
                    }
                )

            # Second call to get the final grounded answer using tool output.
            # Enforce the rigorous tool synthesis grounding contract.
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                messages[0]["content"] = TOOL_SYNTHESIS_SYSTEM_PROMPT

            completion_args["messages"] = messages
            if "tools" in completion_args:
                del completion_args["tools"]
            if "tool_choice" in completion_args:
                del completion_args["tool_choice"]
            if completion_args.get("max_completion_tokens", 0) < 600:
                completion_args["max_completion_tokens"] = 600

            final_response = make_llm_call(completion_args)
            final_answer = final_response.choices[0].message.content or ""

            if return_tool_info:
                return {
                    "answer": final_answer,
                    "tool_used": True,
                    "tool_name": executed_name,
                    "tool_query": executed_query,
                    "tool_sources": collected_sources,
                }
            return final_answer

        final_answer = message.content or ""
        if return_tool_info:
            return {
                "answer": final_answer,
                "tool_used": False,
                "tool_name": None,
                "tool_query": None,
                "tool_sources": [],
            }
        return final_answer

    except ServiceUnavailableError:
        raise LLMServiceError(
            "AI service is temporarily unavailable. Please try again."
        )


class OmniRouteChatModel(BaseChatModel):
    """LangChain ChatModel adapter targeting OmniRoute's OpenAI-compatible gateway."""

    model_name: str
    base_url: str = "http://127.0.0.1:20128/v1"
    api_key: str = "sk-dummy"
    temperature: float = 0.0
    max_tokens: int = 500

    @property
    def _llm_type(self) -> str:
        return "omniroute-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        litellm_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                litellm_messages.append({"role": "system", "content": str(m.content)})
            elif isinstance(m, HumanMessage):
                litellm_messages.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                litellm_messages.append({"role": "assistant", "content": str(m.content)})
            else:
                litellm_messages.append({"role": "user", "content": str(m.content)})

        model_identifier = (
            self.model_name
            if self.model_name.startswith("openai/")
            else f"openai/{self.model_name}"
        )
        response = make_llm_call(
            {
                "model": model_identifier,
                "api_base": self.base_url,
                "api_key": self.api_key or "sk-dummy",
                "messages": litellm_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": 30,
            }
        )
        content = response.choices[0].message.content or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def with_structured_output(self, schema: Any, **kwargs: Any):
        schema_json = json.dumps(schema.model_json_schema())
        schema_instruction = (
            f"\n\nIMPORTANT: Return ONLY a valid JSON object matching this schema with all required fields:\n{schema_json}\nDo not include any Markdown fences or text outside JSON."
        )

        def _invoke_structured(input_val: Any) -> Any:
            messages = input_val.to_messages() if hasattr(input_val, "to_messages") else input_val
            augmented_messages = list(messages)
            if augmented_messages and isinstance(augmented_messages[0], SystemMessage):
                augmented_messages[0] = SystemMessage(content=augmented_messages[0].content + schema_instruction)
            else:
                augmented_messages.insert(0, SystemMessage(content=schema_instruction))

            res = self.invoke(augmented_messages)
            content = str(res.content)
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            return schema.model_validate_json(content)

        return RunnableLambda(_invoke_structured)


def create_chat_model_from_route(route: dict) -> BaseChatModel:
    """Create the LangChain model used by composed workflows such as RAG."""
    base_url, api_key = get_omniroute_config()
    model_str = route.get("model", "harshu-general")
    role_name = model_str.split("/", 1)[-1] if "/" in model_str else model_str

    return OmniRouteChatModel(
        model_name=role_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=route.get("max_tokens", 500),
    )

