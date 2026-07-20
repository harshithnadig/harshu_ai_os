from harshu_ai_os.rag.prompt import build_grounded_prompt
from harshu_ai_os.rag.retriever import retrieve_best_note


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
