"""Google embedding client and the shared text-to-vector boundary."""

from dotenv import load_dotenv
from google import genai


def get_embedding_client():
    """Create the provider client after loading local environment settings."""
    load_dotenv()
    return genai.Client()


def embed_text(client, text: str) -> list[float]:
    """Use one embedding model for both stored chunks and incoming questions."""
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
    )

    return response.embeddings[0].values
