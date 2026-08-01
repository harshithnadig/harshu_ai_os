"""Script to ingest all text files in examples/documents into Chroma DB."""

from pathlib import Path
from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.ingestion import ingest_document

def main():
    collection = get_notes_collection()
    client = get_embedding_client()
    docs_dir = Path("examples/documents")

    for file_path in docs_dir.glob("*.txt"):
        print(f"Ingesting {file_path.name}...")
        chunk_ids = ingest_document(collection, client, file_path, chunk_size=30)
        print(f"Ingested {len(chunk_ids)} chunks from {file_path.name}")

if __name__ == "__main__":
    main()
