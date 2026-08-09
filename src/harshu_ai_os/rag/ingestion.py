"""Document loading, fixed-word chunking, and Chroma ingestion."""

from pathlib import Path

from harshu_ai_os.rag.chroma_store import upsert_chunk_records


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
) -> list[dict]:
    """Give each chunk a stable ID and source metadata before storing it."""
    chunks = load_and_chunk_document(path, chunk_size)
    records = []

    # enumerate gives each chunk its position without a manual counter.
    for index, chunk in enumerate(chunks):
        records.append(
            {
                "id": f"{path.stem}-{index}",
                "text": chunk,
                "source": path.name,
                "chunk_index": index,
            }
        )

    return records


def ingest_document(
    collection,
    client,
    path: Path,
    chunk_size: int,
) -> list[str]:
    """Run the complete local document-to-Chroma ingestion path."""
    records = build_chunk_records(path, chunk_size)

    return upsert_chunk_records(
        collection,
        client,
        records,
    )
