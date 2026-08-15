"""OmniRoute embedding client and the shared text-to-vector boundary."""

import httpx
from harshu_ai_os.core import get_omniroute_config

EMBEDDING_MODEL = "harshu-embedding"


def get_embedding_client():
    """Create the gateway embedding client configured with base URL and authorization."""
    base_url, api_key = get_omniroute_config()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return httpx.Client(base_url=base_url, headers=headers, timeout=20.0)


def embed_text(client, text: str) -> list[float]:
    """Turn text into meaning-based vector embeddings via the OmniRoute gateway."""
    if hasattr(client, "models"):
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values

    if hasattr(client, "post"):
        response = client.post(
            "/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload["data"][0]["embedding"]

    raise ValueError("Client must support HTTP post or embed_content.")


