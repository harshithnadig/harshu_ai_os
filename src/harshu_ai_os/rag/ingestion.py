from pathlib import Path
from harshu_ai_os.rag.chroma_store import upsert_chunk_records
from harshu_ai_os.rag.chunker import chunk_text

def load_and_chunk_document(
    path: Path,
    chunk_size: int,
) -> list[str]:

    if not path.exists():
        raise ValueError(f"Document not found at {path}")
    
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, chunk_size)

def build_chunk_records(
    path: Path,
    chunk_size: int,
) -> list[dict]:

    chunks = load_and_chunk_document(path, chunk_size)
    records = []

    for index, chunk in enumerate(chunks):
        records.append({
            "id": f"{path.stem}-{index}",
            "text": chunk,
            "source": path.name,
            "chunk_index": index,
        })
    return records

def ingest_document(
    collection,
    client,
    path: Path,
    chunk_size: int,
) -> list[str]:

    records = build_chunk_records(path, chunk_size)

    return upsert_chunk_records(
        collection,
        client,
        records,
    )