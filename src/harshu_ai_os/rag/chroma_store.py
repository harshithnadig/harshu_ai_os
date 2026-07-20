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