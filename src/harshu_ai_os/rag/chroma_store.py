from pathlib import Path
from harshu_ai_os.rag.retriever import embed_text

import chromadb


DEFAULT_CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "harshu_ai_os_notes"


def get_notes_collection(path: Path = DEFAULT_CHROMA_PATH):
    client = chromadb.PersistentClient(path=str(path))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

def upsert_notes(collection, client, notes):
    if not notes:
        raise ValueError("At least one note is required.")

    ids = []
    embeddings = []
    metadatas = []

    for index, note in enumerate(notes):
        ids.append(f"note-{index}")
        embeddings.append(embed_text(client, note))
        metadatas.append(
            {
                "source": "manual",
                "position": index,
            }
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=notes,
        metadatas=metadatas,
    )

    return ids

def query_notes(collection, client, question):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    question_embedding = embed_text(client, question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
    )
    
    if not results["documents"] or not results["documents"][0]:
        raise ValueError("No matching notes found.")

    return {
        "ids": results["ids"][0],
        "texts": results["documents"][0],
        "distances": results["distances"][0],
        "metadatas": results["metadatas"][0],
    }
