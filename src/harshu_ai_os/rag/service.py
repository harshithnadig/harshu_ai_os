from harshu_ai_os.rag.prompt import build_grounded_prompt
from harshu_ai_os.rag.retriever import retrieve_best_note
from harshu_ai_os.rag.chroma_store import query_notes


def answer_with_rag(client, question, notes, generate_text):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieval = retrieve_best_note(
        client,
        question,
        notes,
    )

    prompt = build_grounded_prompt(
        question,
        retrieval["text"],
    )

    answer = generate_text(prompt)

    return {
        "answer": answer,
        "context": retrieval["text"],
        "score": retrieval["score"],
        "index": retrieval["index"],
    }

def answer_with_chroma_rag(
    collection,
    client,
    question,
    generate_text,
):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieval = query_notes(collection, client, question)
    citations = build_citations(retrieval)
    context = "\n\n".join(retrieval["texts"])
    prompt = build_grounded_prompt(
        question,
        context,
    )

    answer = generate_text(prompt)

    return {
        "answer": answer,
        "context": context,
        "distances": retrieval["distances"],
        "ids": retrieval["ids"],
        "metadatas": retrieval["metadatas"],
        "citations": citations,
    }

def build_citations(retrieval: dict) -> list[dict]:
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