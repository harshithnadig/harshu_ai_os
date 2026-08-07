"""Run the RAG workflow: retrieve, judge, answer, and cite evidence.

Beginner map:
    question -> retrieve chunks -> check evidence -> generate answer -> cite chunks

This module coordinates the workflow. The smaller modules still own their
specialised jobs: Chroma owns retrieval, the sufficiency judge owns the
evidence decision, and the LLM client owns provider setup.
"""

from time import perf_counter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from harshu_ai_os.llm.client import create_chat_model_from_route
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.rag.chroma_store import query_notes
from harshu_ai_os.rag.sufficiency_judge import judge_context_sufficiency


# Chroma cosine distance is smaller when two embeddings are more similar.
# This starting threshold must be tuned with real evaluation cases, not guessed.
DEFAULT_MAXIMUM_DISTANCE = 0.5
ABSTENTION_ANSWER = "I do not have enough information."


def create_grounded_chat_prompt() -> ChatPromptTemplate:
    """Keep trusted instructions separate from untrusted retrieved text."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer the question using only the supplied context.
Use all supplied passages together when the conclusion is directly supported.
If the context does not contain the answer, say:
I do not have enough information.""",
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
    """Connect the prompt, selected model, and plain-text output parser."""
    return create_grounded_chat_prompt() | model | StrOutputParser()


def generate_grounded_answer(route: dict, question: str, context: str) -> str:
    """Generate one answer from context and expose one stable failure message."""
    try:
        model = create_chat_model_from_route(route)
        chain = create_grounded_text_chain(model)
        result = chain.invoke({"context": context, "question": question})
        return str(result)
    except Exception as error:
        # Provider libraries raise different exceptions. The API should not
        # force its caller to understand every provider-specific error type.
        raise LLMServiceError(
            "AI service is temporarily unavailable. Please try again."
        ) from error


def elapsed_ms(started_at: float) -> float:
    """Convert a performance-counter duration into readable milliseconds."""
    return (perf_counter() - started_at) * 1000


def should_abstain(distances: list[float], maximum_distance: float) -> bool:
    """Return True when no retrieved chunk is close enough to trust."""
    # No results means there is no evidence, so the safest action is abstention.
    if not distances:
        return True

    # One sufficiently close chunk is enough to continue to the stricter judge.
    return min(distances) > maximum_distance


def build_citations(retrieval: dict) -> list[dict]:
    """Turn retrieval metadata into the public citation shape."""
    citations = []

    # zip keeps each chunk ID beside its own distance and metadata.
    for chunk_id, distance, metadata in zip(
        retrieval["ids"], retrieval["distances"], retrieval["metadatas"]
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


def build_rag_result(
    retrieval: dict,
    *,
    answer: str,
    context: str,
    abstained: bool,
    judge_reason: str,
    citations: list[dict],
    retrieval_ms: float,
    judge_ms: float,
    generation_ms: float,
    total_ms: float,
) -> dict:
    """Build the response once so every RAG exit uses the same fields."""
    return {
        "answer": answer,
        "abstained": abstained,
        "abstention_reason": "insufficient_context" if abstained else None,
        "judge_reason": judge_reason,
        "context": context,
        "distances": retrieval["distances"],
        "ids": retrieval["ids"],
        "metadatas": retrieval["metadatas"],
        "citations": citations,
        "retrieval_texts": retrieval["texts"],
        "retrieval_ms": retrieval_ms,
        "judge_ms": judge_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
    }


def answer_with_chroma_rag(
    collection,
    client,
    question: str,
    route: dict,
    maximum_distance: float = DEFAULT_MAXIMUM_DISTANCE,
) -> dict:
    """Run the complete RAG path while keeping each decision visible."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    total_started_at = perf_counter()

    # Step 1: retrieve candidate evidence from Chroma.
    retrieval_started_at = perf_counter()
    retrieval = query_notes(collection, client, question)
    retrieval_ms = elapsed_ms(retrieval_started_at)
    all_context = "\n\n".join(retrieval["texts"])

    # Step 2: use the cheap distance gate before spending another model call.
    if should_abstain(retrieval["distances"], maximum_distance):
        return build_rag_result(
            retrieval,
            answer=ABSTENTION_ANSWER,
            context=all_context,
            abstained=True,
            judge_reason="Distance filter threshold exceeded.",
            citations=[],
            retrieval_ms=retrieval_ms,
            judge_ms=0.0,
            generation_ms=0.0,
            total_ms=elapsed_ms(total_started_at),
        )

    # Step 3: ask the typed judge whether the chunks answer the full question.
    judge_started_at = perf_counter()
    verdict = judge_context_sufficiency(
        route, question, retrieval["texts"], retrieval["ids"]
    )
    judge_ms = elapsed_ms(judge_started_at)

    # Never trust invented chunk IDs returned by a model.
    known_ids = set(retrieval["ids"])
    supporting_ids = [
        chunk_id
        for chunk_id in verdict.supporting_chunk_ids
        if chunk_id in known_ids
    ]
    verdict_is_valid = (
        not verdict.answerable and not verdict.supporting_chunk_ids
    ) or (verdict.answerable and bool(supporting_ids))

    if not verdict.answerable or not verdict_is_valid:
        reason = (
            verdict.reason
            if verdict_is_valid
            else "Sufficiency judge returned an invalid verdict."
        )
        return build_rag_result(
            retrieval,
            answer=ABSTENTION_ANSWER,
            context=all_context,
            abstained=True,
            judge_reason=reason,
            citations=[],
            retrieval_ms=retrieval_ms,
            judge_ms=judge_ms,
            generation_ms=0.0,
            total_ms=elapsed_ms(total_started_at),
        )

    # Step 4: generate from supporting chunks only, not every retrieved chunk.
    supporting_indices = [
        index
        for index, chunk_id in enumerate(retrieval["ids"])
        if chunk_id in supporting_ids
    ]
    supporting_context = "\n\n".join(
        retrieval["texts"][index] for index in supporting_indices
    )

    generation_started_at = perf_counter()
    answer = generate_grounded_answer(route, question, supporting_context)
    generation_ms = elapsed_ms(generation_started_at)

    # Citations must describe only the evidence used to generate the answer.
    citations = [
        citation
        for citation in build_citations(retrieval)
        if citation["chunk_id"] in supporting_ids
    ]

    return build_rag_result(
        retrieval,
        answer=answer,
        context=supporting_context,
        abstained=False,
        judge_reason=verdict.reason,
        citations=citations,
        retrieval_ms=retrieval_ms,
        judge_ms=judge_ms,
        generation_ms=generation_ms,
        total_ms=elapsed_ms(total_started_at),
    )
