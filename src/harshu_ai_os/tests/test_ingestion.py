import pytest

from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.ingestion import (
    build_chunk_records,
    chunk_text,
    find_indexed_document_state,
    hash_document,
    remove_document_chunks,
    sync_document,
    sync_knowledge_base,
    sync_one_document,
)


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResponse:
    def __init__(self, values):
        self.embeddings = [FakeEmbedding(values)]


class FakeModels:
    def __init__(self):
        self.call_count = 0

    def embed_content(self, model, contents):
        self.call_count += 1
        return FakeResponse([1.0, 0.0])


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_chunk_text_splits_text_into_word_groups():
    result = chunk_text(
        "one two three four five six seven",
        3,
    )

    assert result == [
        "one two three",
        "four five six",
        "seven",
    ]


def test_build_chunk_records_adds_source_and_chunk_ids(tmp_path):
    document_path = tmp_path / "project_notes.txt"
    document_path.write_text(
        "one two three four five",
        encoding="utf-8",
    )

    records = build_chunk_records(
        document_path,
        chunk_size=2,
    )

    assert records == [
        {
            "id": "project_notes-0",
            "text": "one two",
            "source": "project_notes.txt",
            "chunk_index": 0,
        },
        {
            "id": "project_notes-1",
            "text": "three four",
            "source": "project_notes.txt",
            "chunk_index": 1,
        },
        {
            "id": "project_notes-2",
            "text": "five",
            "source": "project_notes.txt",
            "chunk_index": 2,
        },
    ]


def test_hash_document_deterministic(tmp_path):
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("hello world", encoding="utf-8")

    hash1 = hash_document(doc_path)
    hash2 = hash_document(doc_path)
    assert hash1 == hash2
    assert len(hash1) == 64

    doc_path.write_text("hello world updated", encoding="utf-8")
    assert hash_document(doc_path) != hash1


def test_hash_document_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Document not found"):
        hash_document(tmp_path / "non_existent.txt")


def test_sync_new_document(tmp_path):
    """Test A: New document is chunked, embedded, and inserted into Chroma with hash."""
    collection = get_notes_collection(tmp_path / "chroma")
    client = FakeClient()

    doc_path = tmp_path / "new_doc.txt"
    doc_path.write_text("alpha beta gamma delta", encoding="utf-8")

    res = sync_one_document(collection, client, doc_path, chunk_size=2)

    assert res["status"] == "created"
    assert res["reason"] == "new"
    assert res["chunks_indexed"] == 2
    assert res["chunks_deleted"] == 0
    assert client.models.call_count == 2

    state = find_indexed_document_state(collection)
    assert "new_doc.txt" in state
    assert state["new_doc.txt"]["doc_hash"] == hash_document(doc_path, chunk_size=2)
    assert len(state["new_doc.txt"]["chunk_ids"]) == 2


def test_sync_unchanged_document_skipped(tmp_path):
    """Test B: Unchanged document skips embedding and upsert."""
    collection = get_notes_collection(tmp_path / "chroma")
    client = FakeClient()

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("one two three four", encoding="utf-8")

    # First sync
    res1 = sync_one_document(collection, client, doc_path, chunk_size=2)
    assert res1["status"] == "created"
    initial_call_count = client.models.call_count
    assert initial_call_count == 2

    # Second sync without changes
    res2 = sync_one_document(collection, client, doc_path, chunk_size=2)
    assert res2["status"] == "skipped"
    assert res2["reason"] == "unchanged"
    assert res2["chunks_indexed"] == 0
    assert res2["chunks_deleted"] == 0

    # Verify 0 additional embedding calls were made
    assert client.models.call_count == initial_call_count


def test_sync_changed_document_replaces_old_chunks(tmp_path):
    """Test C: Changed document deletes old chunks and inserts fresh chunks."""
    collection = get_notes_collection(tmp_path / "chroma")
    client = FakeClient()

    doc_path = tmp_path / "notes.txt"
    doc_path.write_text("initial word one two", encoding="utf-8")

    sync_document(collection, client, doc_path, chunk_size=2)

    # Change document content
    doc_path.write_text("modified word three four", encoding="utf-8")

    res = sync_document(collection, client, doc_path, chunk_size=2)
    assert res["status"] == "updated"
    assert res["reason"] == "changed"
    assert res["chunks_indexed"] == 2
    assert res["chunks_deleted"] == 2

    data = collection.get(include=["documents", "metadatas"])
    assert "initial word" not in data["documents"]
    assert "modified word" in data["documents"]
    assert data["metadatas"][0]["doc_hash"] == hash_document(doc_path, chunk_size=2)


def test_sync_shortened_document_leaves_no_ghost_chunks(tmp_path):
    """Test D: Shortened document leaves no leftover/ghost chunks in Chroma."""
    collection = get_notes_collection(tmp_path / "chroma")
    client = FakeClient()

    doc_path = tmp_path / "long_doc.txt"
    doc_path.write_text(
        "one two three four five six seven eight nine ten",
        encoding="utf-8",
    )

    # Initial sync with chunk_size=2 produces 5 chunks (long_doc-0 to long_doc-4)
    res1 = sync_one_document(collection, client, doc_path, chunk_size=2)
    assert res1["chunks_indexed"] == 5
    assert collection.count() == 5

    # Shorten document to only 2 words (1 chunk)
    doc_path.write_text("short document", encoding="utf-8")

    res2 = sync_one_document(collection, client, doc_path, chunk_size=2)
    assert res2["status"] == "updated"
    assert res2["chunks_indexed"] == 1
    assert res2["chunks_deleted"] == 5

    # Verify no ghost chunks remain in Chroma
    assert collection.count() == 1
    stored = collection.get()
    assert stored["ids"] == ["long_doc-0"]
    assert stored["documents"] == ["short document"]


def test_sync_knowledge_base_deleted_source_removes_indexed_chunks(tmp_path):
    """Test E: Deleted source file causes its chunks to be removed during knowledge base sync."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    chroma_dir = tmp_path / "chroma"

    doc_a = docs_dir / "doc_a.txt"
    doc_b = docs_dir / "doc_b.txt"
    doc_a.write_text("document a content words here", encoding="utf-8")
    doc_b.write_text("document b content words here", encoding="utf-8")

    collection = get_notes_collection(chroma_dir)
    client = FakeClient()

    # Initial sync of both documents
    res1 = sync_knowledge_base(collection, client, docs_dir, chunk_size=2)
    assert set(res1["synced"]) == {"doc_a.txt", "doc_b.txt"}
    assert res1["deleted"] == []
    assert res1["skipped"] == []

    # Verify both documents have chunks in Chroma
    data1 = collection.get(where={"source": "doc_b.txt"})
    assert len(data1["ids"]) > 0

    # Delete doc_b.txt from filesystem
    doc_b.unlink()

    # Re-sync knowledge base
    res2 = sync_knowledge_base(collection, client, docs_dir, chunk_size=2)
    assert res2["synced"] == []
    assert res2["skipped"] == ["doc_a.txt"]
    assert res2["deleted"] == ["doc_b.txt"]

    # Verify doc_b.txt chunks were purged from Chroma
    data_b = collection.get(where={"source": "doc_b.txt"})
    assert data_b["ids"] == []

    # Verify doc_a.txt chunks are still intact
    data_a = collection.get(where={"source": "doc_a.txt"})
    assert len(data_a["ids"]) > 0


def test_remove_document_chunks_standalone(tmp_path):
    collection = get_notes_collection(tmp_path / "chroma")
    client = FakeClient()

    doc_path = tmp_path / "to_delete.txt"
    doc_path.write_text("temp data", encoding="utf-8")

    sync_one_document(collection, client, doc_path, chunk_size=2)
    assert collection.count() == 1

    deleted_ids = remove_document_chunks(collection, "to_delete.txt")
    assert deleted_ids == ["to_delete-0"]
    assert collection.count() == 0


def test_sync_changed_chunk_size_triggers_rebuild(tmp_path):
    """Test F: Changing chunk_size without changing text triggers rebuild rather than skipping."""
    collection = get_notes_collection(tmp_path / "chroma")
    client = FakeClient()

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("one two three four five six", encoding="utf-8")

    # 1. Sync with chunk_size=2 -> 3 chunks (doc-0, doc-1, doc-2)
    res1 = sync_one_document(collection, client, doc_path, chunk_size=2)
    assert res1["status"] == "created"
    assert res1["chunks_indexed"] == 3
    assert collection.count() == 3
    data1 = collection.get()
    assert sorted(data1["ids"]) == ["doc-0", "doc-1", "doc-2"]
    assert client.models.call_count == 3

    # 2. Sync again with chunk_size=3 without changing document text
    res2 = sync_one_document(collection, client, doc_path, chunk_size=3)
    assert res2["status"] == "updated"
    assert res2["reason"] == "changed"
    assert res2["chunks_indexed"] == 2
    assert res2["chunks_deleted"] == 3

    # 3. Prove old chunks are removed and new chunk layout (2 chunks: doc-0, doc-1) is stored
    assert collection.count() == 2
    data2 = collection.get()
    assert sorted(data2["ids"]) == ["doc-0", "doc-1"]
    assert "doc-2" not in data2["ids"]
    assert data2["documents"] == ["one two three", "four five six"]
    assert data2["metadatas"][0]["doc_hash"] == hash_document(doc_path, chunk_size=3)
    assert client.models.call_count == 5

