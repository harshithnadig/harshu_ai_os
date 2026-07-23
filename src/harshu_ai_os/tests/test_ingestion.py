from harshu_ai_os.rag.ingestion import build_chunk_records


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