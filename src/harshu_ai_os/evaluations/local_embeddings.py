"""
Local embedding wrapper for benchmark arena to bypass Gemini API limits.
Uses sentence-transformers to generate fast, high-quality local embeddings.
"""

from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# We use nomic-ai/nomic-embed-text-v1.5 as requested
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

_model = None

def get_local_model():
    """Lazy load the sentence transformer model."""
    global _model
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. Run: pip install sentence-transformers")
    
    if _model is None:
        print(f"Loading local embedding model: {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts locally."""
    model = get_local_model()
    # Nomic expects "search_document: " for documents
    prefixed_texts = ["search_document: " + t for t in texts]
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True)
    return embeddings.tolist()

def embed_query(text: str) -> List[float]:
    """Embed a single query locally."""
    model = get_local_model()
    # Nomic expects "search_query: " for queries
    instruction = "search_query: "
    embedding = model.encode(instruction + text, normalize_embeddings=True)
    return embedding.tolist()
