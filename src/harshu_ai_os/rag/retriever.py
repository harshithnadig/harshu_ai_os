import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def embed_text(client, text):
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
    )

    return response.embeddings[0].values


def retrieve_best_note(client, question, notes):
    if not notes:
        raise ValueError("At least one note is required.")
    question_embedding = embed_text(client, question)

    note_embeddings = []

    for note in notes:
        note_embedding = embed_text(client, note)
        note_embeddings.append(note_embedding)

    question_vector = np.array(question_embedding).reshape(1, -1)
    note_vectors = np.array(note_embeddings)

    similarity_scores = cosine_similarity(
        question_vector,
        note_vectors,
    )[0]

    best_index = int(np.argmax(similarity_scores))
    best_note = notes[best_index]
    best_score = float(similarity_scores[best_index])

    return {
        "text": best_note,
        "score": best_score,
        "index": best_index,
    }
