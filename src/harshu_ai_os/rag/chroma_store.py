from pathlib import Path

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