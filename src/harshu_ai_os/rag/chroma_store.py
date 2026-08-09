"""Persistent Chroma storage and semantic retrieval operations."""

from pathlib import Path

import chromadb

from harshu_ai_os.rag.embedding_client import embed_text


DEFAULT_CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "harshu_ai_os_notes"
DEFAULT_TOP_K = 5


def get_notes_collection(path: Path = DEFAULT_CHROMA_PATH):
    """Open the one persistent local collection used by the application."""
    client = chromadb.PersistentClient(path=str(path))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )


def query_notes(collection, client, question, top_k: int = DEFAULT_TOP_K):
    """Embed one question and return its closest stored text chunks."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    question_embedding = embed_text(client, question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    if not results["documents"] or not results["documents"][0]:
        raise ValueError("No matching notes found.")

    # Chroma nests results because it can accept several questions at once.
    return {
        "ids": results["ids"][0],
        "texts": results["documents"][0],
        "distances": results["distances"][0],
        "metadatas": results["metadatas"][0],
    }


def upsert_chunk_records(collection, client, records):
    """Persist document chunks with the metadata needed for later citations."""
    if not records:
        raise ValueError("At least one chunk record is required.")

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for record in records:
        ids.append(record["id"])
        documents.append(record["text"])
        embeddings.append(embed_text(client, record["text"]))
        metadatas.append(
            {
                "source": record["source"],
                "chunk_index": record["chunk_index"],
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return ids
