"""Helper to register clean API keys into the running OmniRoute instance."""

import re
import sys
from pathlib import Path
import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"

client = httpx.Client(timeout=15.0)
login_res = client.post("http://127.0.0.1:20128/api/auth/login", json={"password": "CHANGEME"})
if login_res.status_code != 200:
    print(f"Login failed: {login_res.status_code} {login_res.text}")
    sys.exit(1)

# Delete existing provider connections to avoid duplicates
existing = client.get("http://127.0.0.1:20128/api/providers").json().get("connections", [])
for conn in existing:
    client.request("DELETE", "http://127.0.0.1:20128/api/providers", json={"connectionIds": [conn["id"]]})

# Read and clean environment variables
env_vars = {}
if ENV_FILE.exists():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                clean_k = k.strip()
                clean_v = v.strip().strip('"').strip("'")
                env_vars[clean_k] = clean_v

providers_to_register = [
    ("groq", "GROQ_API_KEY", "Groq Main"),
    ("gemini", "GEMINI_API_KEY", "Google Gemini Main"),
    ("cerebras", "CEREBRAS_API_KEY", "Cerebras Main"),
    ("cohere", "COHERE_API_KEY", "Cohere Main"),
]

for prov_id, env_key, name in providers_to_register:
    key_val = env_vars.get(env_key)
    if key_val:
        payload = {"provider": prov_id, "name": name, "apiKey": key_val}
        res = client.post("http://127.0.0.1:20128/api/providers", json=payload)
        print(f"Registered {prov_id} ({env_key}): HTTP {res.status_code}")
    else:
        print(f"Skipped {prov_id} (No {env_key} in .env)")

print("Provider registration complete.")
