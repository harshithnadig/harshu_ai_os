from pathlib import Path

from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.ingestion import ingest_document


collection = get_notes_collection()
client = get_embedding_client()
document_path = Path("examples/documents/harshu_ai_os_overview.txt")

ids = ingest_document(
    collection,
    client,
    document_path,
    chunk_size=50,
)

print("Ingested document:", document_path)
print("Stored chunk IDs:", ids)
print("Total notes:", collection.count())
