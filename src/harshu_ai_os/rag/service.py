"""The complete retrieval, grounded generation, and citation workflow."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from harshu_ai_os.llm.client import create_chat_model_from_route
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.rag.chroma_store import query_notes


def create_grounded_chat_prompt() -> ChatPromptTemplate:
    """Build the fixed grounding rules separately from retrieved user data."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer the question using only the supplied context.
Use all supplied passages together when the conclusion is directly supported.
If the context does not contain the answer, say:
"I do not have enough information.\"""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}""",
            ),
        ]
    )


def create_grounded_text_chain(model):
    """Compose the RAG-only prompt, selected model, and text boundary."""
    return create_grounded_chat_prompt() | model | StrOutputParser()


def generate_grounded_answer(
    route: dict,
    question: str,
    context: str,
) -> str:
    """Run the provider-neutral LangChain generation step for retrieved context."""
    try:
        model = create_chat_model_from_route(route)
        chain = create_grounded_text_chain(model)
        result = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )
        return str(result)
    except Exception as error:
        # The API exposes one stable provider-failure contract across integrations.
        raise LLMServiceError(
            "AI service is temporarily unavailable. Please try again."
        ) from error


def answer_with_chroma_rag(
    collection,
    client,
    question,
    route,
):
    """Retrieve evidence, generate one grounded answer, and return its trace data."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieval = query_notes(collection, client, question)
    citations = build_citations(retrieval)
    context = "\n\n".join(retrieval["texts"])
    answer = generate_grounded_answer(route, question, context)

    return {
        "answer": answer,
        "context": context,
        "distances": retrieval["distances"],
        "ids": retrieval["ids"],
        "metadatas": retrieval["metadatas"],
        "citations": citations,
    }


def build_citations(retrieval: dict) -> list[dict]:
    """Keep retrieval provenance beside the generated answer."""
    citations = []

    for chunk_id, distance, metadata in zip(
        retrieval["ids"],
        retrieval["distances"],
        retrieval["metadatas"],
    ):
        citations.append(
            {
                "source": metadata["source"],
                "chunk_id": chunk_id,
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    return citations


def should_abstain(
    distances: list[float],
    maximum_distance: float,
) -> bool:
    """Return True when no retrieved chunk meets the configured distance limit."""
    if not distances:
        return True
    if min(distances) > maximum_distance:
        return True
    return False
