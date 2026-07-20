from harshu_ai_os.rag.chroma_store import (
    COLLECTION_NAME,
    get_notes_collection,
)


def test_get_notes_collection_creates_empty_collection(tmp_path):
    collection = get_notes_collection(tmp_path)

    assert collection.name == COLLECTION_NAME
    assert collection.count() == 0
