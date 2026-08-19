"""Sync bundled example documents into the local Chroma collection."""

from pathlib import Path

from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.ingestion import sync_knowledge_base


def main() -> None:
    """Make one explicit sync pass through the bundled synthetic documents."""
    collection = get_notes_collection()
    client = get_embedding_client()
    result = sync_knowledge_base(
        collection,
        client,
        Path("examples/documents"),
        chunk_size=50,
    )

    print(
        f"Knowledge sync complete: {len(result['synced'])} synced, "
        f"{len(result['skipped'])} skipped, {len(result['deleted'])} deleted. "
        f"Total indexed chunks: {collection.count()}"
    )


if __name__ == "__main__":
    main()

