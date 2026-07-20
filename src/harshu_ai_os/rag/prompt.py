def build_grounded_prompt(question, context):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    return f"""
    Answer the question using only the provided context. If the context does not contain the answer, say that the available context is insufficient.

Context:
{context}

Question:
{question}
"""