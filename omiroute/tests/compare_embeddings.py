import os
import httpx
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from harshu_ai_os.rag.embedding_client import get_embedding_client, embed_text

load_dotenv()

# 1. Direct embedding from Google GenAI SDK
text = "Harshu AI OS local embedding comparison"
direct_client = get_embedding_client()
direct_vec = embed_text(direct_client, text)
print(f"Direct Google SDK embedding:")
print(f"  Model: gemini-embedding-2")
print(f"  Vector length: {len(direct_vec)}")
print(f"  First 5 floats: {direct_vec[:5]}")

# 2. OmniRoute gateway embedding
env_file = Path(__file__).resolve().parent.parent / ".env"
token = ""
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("OMNIROUTE_API_KEY="):
                token = line.split("=", 1)[1].strip()

r = httpx.post(
    "http://127.0.0.1:20128/v1/embeddings",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={"model": "harshu-embedding", "input": text},
    timeout=15.0,
)
data = r.json()
gateway_vec = data["data"][0]["embedding"]
gateway_model = data.get("model")
print(f"\nOmniRoute Gateway embedding:")
print(f"  Logical Model: harshu-embedding")
print(f"  Returned Model Identity: {gateway_model}")
print(f"  Vector length: {len(gateway_vec)}")
print(f"  First 5 floats: {gateway_vec[:5]}")

# Compare
print(f"\nComparison:")
print(f"  Dimension Match: {len(direct_vec) == len(gateway_vec)} ({len(direct_vec)} == {len(gateway_vec)})")
# Check cosine similarity
import numpy as np
dot = np.dot(direct_vec, gateway_vec)
norm_a = np.linalg.norm(direct_vec)
norm_b = np.linalg.norm(gateway_vec)
cos_sim = dot / (norm_a * norm_b)
print(f"  Cosine Similarity between Direct and Gateway: {round(float(cos_sim), 6)}")
