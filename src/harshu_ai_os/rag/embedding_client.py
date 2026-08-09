"""Google embedding client and the shared text-to-vector boundary."""

from dotenv import load_dotenv
from google import genai

EMBEDDING_MODEL = "gemini-embedding-2"


def get_embedding_client():
    """Create the provider client after loading local environment settings."""
    load_dotenv()
    return genai.Client()


def embed_text(client, text: str) -> list[float]:
    """Turn text into meaning-based numbers used for similarity search.

    Stored chunks and incoming questions must use the same embedding model so
    that Chroma can compare their vectors correctly.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values
