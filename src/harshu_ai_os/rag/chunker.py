def chunk_text(text: str, chunk_size: int) -> list[str]:
    if not text.strip():
        raise ValueError("Text cannot be empty")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    split_words = text.split()
    chunks = []

    for i in range(0, len(split_words), chunk_size):
        chunk = split_words[i:i+chunk_size]
        chunks.append(" ".join(chunk))

    return chunks


