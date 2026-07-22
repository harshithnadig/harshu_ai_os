from harshu_ai_os.rag.chroma_store import (
    get_notes_collection,
    upsert_notes,
)
from harshu_ai_os.rag.embedding_client import get_embedding_client


collection = get_notes_collection()
client = get_embedding_client()

notes = [
    "Harshu AI OS uses FastAPI to expose API endpoints.",
    "The task router classifies a question and selects an appropriate language model.",
    "ChromaDB stores document embeddings and retrieves relevant notes for grounded answers.",
]

ids = upsert_notes(collection, client, notes)

print("Stored note IDs:", ids)
print("Total notes:", collection.count())