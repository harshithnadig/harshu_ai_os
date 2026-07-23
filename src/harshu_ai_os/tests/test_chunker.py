from harshu_ai_os.rag.chunker import chunk_text


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