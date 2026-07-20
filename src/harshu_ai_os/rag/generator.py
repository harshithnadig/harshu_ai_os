from collections.abc import Callable

from harshu_ai_os.llm.client import call_llm


def create_rag_generator(
    route: dict,
    llm_call: Callable[[dict, str], str] = call_llm,
):
    def generate_text(prompt: str) -> str:
        return llm_call(route, prompt)

    return generate_text