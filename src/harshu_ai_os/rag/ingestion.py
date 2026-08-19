"""Document loading, fixed-word chunking, and Chroma ingestion and synchronization."""

import hashlib
from pathlib import Path

from harshu_ai_os.rag.chroma_store import upsert_chunk_records


def hash_document(path: Path, chunk_size: int | None = None) -> str:
    """Calculate a deterministic SHA-256 fingerprint for a document and its indexing settings."""
    if not path.exists():
        raise ValueError(f"Document not found at {path}")

    content = path.read_bytes()
    hasher = hashlib.sha256(content)
    if chunk_size is not None:
        hasher.update(f":chunk_size={chunk_size}".encode("utf-8"))
    return hasher.hexdigest()


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Create deterministic fixed-word chunks for the current baseline."""
    if not text.strip():
        raise ValueError("Text cannot be empty")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    split_words = text.split()
    chunks = []

    for index in range(0, len(split_words), chunk_size):
        chunk = split_words[index : index + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks


def load_and_chunk_document(
    path: Path,
    chunk_size: int,
) -> list[str]:
    """Read one UTF-8 document and split it with the current chunking policy."""
    if not path.exists():
        raise ValueError(f"Document not found at {path}")

    text = path.read_text(encoding="utf-8")
    return chunk_text(text, chunk_size)


def build_chunk_records(
    path: Path,
    chunk_size: int,
    doc_hash: str | None = None,
) -> list[dict]:
    """Give each chunk a stable ID and source metadata before storing it."""
    chunks = load_and_chunk_document(path, chunk_size)
    records = []

    # enumerate gives each chunk its position without a manual counter.
    for index, chunk in enumerate(chunks):
        record = {
            "id": f"{path.stem}-{index}",
            "text": chunk,
            "source": path.name,
            "chunk_index": index,
        }
        if doc_hash is not None:
            record["doc_hash"] = doc_hash
        records.append(record)

    return records


def find_indexed_document_state(collection) -> dict[str, dict]:
    """Inspect Chroma to discover currently indexed documents, their hashes, and chunk IDs."""
    data = collection.get(include=["metadatas"])
    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []

    state: dict[str, dict] = {}
    for chunk_id, metadata in zip(ids, metadatas):
        if not metadata or "source" not in metadata:
            continue
        source = metadata["source"]
        if source not in state:
            state[source] = {
                "doc_hash": metadata.get("doc_hash"),
                "chunk_ids": [],
            }
        state[source]["chunk_ids"].append(chunk_id)
        if metadata.get("doc_hash") and not state[source]["doc_hash"]:
            state[source]["doc_hash"] = metadata["doc_hash"]

    return state


def remove_document_chunks(
    collection,
    source: str,
    chunk_ids: list[str] | None = None,
) -> list[str]:
    """Delete all chunks in Chroma belonging to a specific source document."""
    if chunk_ids is None:
        existing = collection.get(where={"source": source})
        chunk_ids = existing.get("ids") or []

    if chunk_ids:
        collection.delete(ids=chunk_ids)

    return chunk_ids


def sync_one_document(
    collection,
    client,
    path: Path,
    chunk_size: int = 50,
    indexed_state: dict | None = None,
) -> dict:
    """Synchronize a single source document with Chroma.

    1. Calculate content hash.
    2. If unchanged: skip embedding and upsert.
    3. If changed/new: delete all old chunks, chunk again, embed again, insert fresh chunks.
    """
    if not path.exists():
        raise ValueError(f"Document not found at {path}")

    current_hash = hash_document(path, chunk_size=chunk_size)
    source = path.name

    if indexed_state is None:
        indexed_state = find_indexed_document_state(collection)

    doc_state = indexed_state.get(source)

    if doc_state is not None and doc_state.get("doc_hash") == current_hash:
        return {
            "source": source,
            "status": "skipped",
            "reason": "unchanged",
            "chunk_ids": doc_state.get("chunk_ids", []),
            "chunks_indexed": 0,
            "chunks_deleted": 0,
        }

    # Document is either new or changed. Delete old chunks if any exist.
    old_chunk_ids = doc_state.get("chunk_ids") if doc_state else None
    deleted_ids = remove_document_chunks(collection, source, chunk_ids=old_chunk_ids)

    records = build_chunk_records(path, chunk_size, doc_hash=current_hash)
    inserted_ids = upsert_chunk_records(collection, client, records)

    return {
        "source": source,
        "status": "updated" if doc_state else "created",
        "reason": "changed" if doc_state else "new",
        "chunk_ids": inserted_ids,
        "chunks_indexed": len(inserted_ids),
        "chunks_deleted": len(deleted_ids),
    }


# Friendly alias
sync_document = sync_one_document


def sync_knowledge_base(
    collection,
    client,
    documents: list[Path] | Path,
    chunk_size: int = 50,
    pattern: str = "*.txt",
) -> dict:
    """Keep Chroma aligned with project documents: sync new/changed, delete removed."""
    if isinstance(documents, Path):
        if documents.is_dir():
            doc_paths = sorted(documents.glob(pattern))
        else:
            doc_paths = [documents]
    else:
        doc_paths = list(documents)

    indexed_state = find_indexed_document_state(collection)

    active_sources: set[str] = set()
    synced: list[str] = []
    skipped: list[str] = []
    deleted: list[str] = []
    total_chunks_indexed = 0
    total_chunks_deleted = 0

    for doc_path in doc_paths:
        active_sources.add(doc_path.name)
        res = sync_one_document(
            collection,
            client,
            doc_path,
            chunk_size=chunk_size,
            indexed_state=indexed_state,
        )
        total_chunks_indexed += res["chunks_indexed"]
        total_chunks_deleted += res["chunks_deleted"]

        if res["status"] == "skipped":
            skipped.append(res["source"])
        else:
            synced.append(res["source"])

    # Remove documents that exist in Chroma index but no longer exist in source documents
    for indexed_source, state in indexed_state.items():
        if indexed_source not in active_sources:
            removed_ids = remove_document_chunks(
                collection,
                indexed_source,
                chunk_ids=state.get("chunk_ids"),
            )
            total_chunks_deleted += len(removed_ids)
            deleted.append(indexed_source)

    return {
        "synced": synced,
        "skipped": skipped,
        "deleted": deleted,
        "total_active": len(active_sources),
        "total_chunks_indexed": total_chunks_indexed,
        "total_chunks_deleted": total_chunks_deleted,
    }


def ingest_document(
    collection,
    client,
    path: Path,
    chunk_size: int,
) -> list[str]:
    """Run the complete local document-to-Chroma ingestion path."""
    res = sync_one_document(collection, client, path, chunk_size)
    return res["chunk_ids"]

