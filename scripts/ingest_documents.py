"""Index every safe example document into the local Chroma collection."""

from pathlib import Path

from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.ingestion import ingest_document


def main() -> None:
    """Make one explicit pass through the bundled synthetic documents."""
    collection = get_notes_collection()
    client = get_embedding_client()
    documents = sorted(Path("examples/documents").glob("*.txt"))

    for document in documents:
        chunk_ids = ingest_document(collection, client, document, chunk_size=50)
        print(f"Indexed {len(chunk_ids)} chunks from {document.name}")

    print(f"Total indexed chunks: {collection.count()}")


if __name__ == "__main__":
    main()
