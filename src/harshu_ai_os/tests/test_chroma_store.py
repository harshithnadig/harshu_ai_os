from harshu_ai_os.rag.chroma_store import (
    COLLECTION_NAME,
    get_notes_collection,
)
from harshu_ai_os.rag.chroma_store import (
    COLLECTION_NAME,
    get_notes_collection,
    upsert_notes,
)

def test_get_notes_collection_creates_empty_collection(tmp_path):
    collection = get_notes_collection(tmp_path)

    assert collection.name == COLLECTION_NAME
    assert collection.count() == 0

class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResponse:
    def __init__(self, values):
        self.embeddings = [FakeEmbedding(values)]


class FakeModels:
    def embed_content(self, model, contents):
        vectors = {
            "FastAPI exposes the endpoint.": [0.0, 1.0],
            "The router selects a model.": [1.0, 0.0],
        }

        return FakeResponse(vectors[contents])


class FakeClient:
    def __init__(self):
        self.models = FakeModels()

def test_upsert_notes_stores_documents_and_metadata(tmp_path):
    collection = get_notes_collection(tmp_path)
    client = FakeClient()

    notes = [
        "FastAPI exposes the endpoint.",
        "The router selects a model.",
    ]

    ids = upsert_notes(
        collection,
        client,
        notes,
    )

    stored = collection.get(
        include=["documents", "metadatas"],
    )

    assert ids == ["note-0", "note-1"]
    assert collection.count() == 2
    assert set(stored["documents"]) == set(notes)
    assert {
        metadata["source"]
        for metadata in stored["metadatas"]
    } == {"manual"}