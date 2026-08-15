import json
import re
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from litellm import completion

env_file = Path(__file__).resolve().parent.parent / ".env"
token = ""
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("OMNIROUTE_API_KEY="):
                token = line.split("=", 1)[1].strip()

class OmniRouteChatModel(BaseChatModel):
    model_name: str
    base_url: str = "http://127.0.0.1:20128/v1"
    api_key: str = token
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

        response = completion(
            model=f"openai/{self.model_name}",
            api_base=self.base_url,
            api_key=self.api_key or "sk-dummy",
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=30,
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

# 1. Plain text chain
chat = OmniRouteChatModel(model_name="harshu-general")
prompt = ChatPromptTemplate.from_messages([("human", "What is 2+2 in one word?")])
chain = prompt | chat | StrOutputParser()
res = chain.invoke({})
print("Plain text chain output:", res)

# 2. Structured output
from harshu_ai_os.rag.sufficiency_judge import create_sufficiency_judge_prompt

class SufficiencyVerdict(BaseModel):
    answerable: bool
    reason: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)

judge = OmniRouteChatModel(model_name="harshu-judge")
structured = judge.with_structured_output(SufficiencyVerdict)
prompt_judge = create_sufficiency_judge_prompt()
prompt_val = prompt_judge.invoke({
    "question": "What is Harshu AI OS?",
    "formatted_chunks": '<chunk id="c1">\nHarshu AI OS is a local AI platform.\n</chunk>'
})
verdict = structured.invoke(prompt_val)
print("Structured verdict success:", verdict)

