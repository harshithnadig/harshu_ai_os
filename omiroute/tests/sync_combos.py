"""Sync combos from config/combos.json to running OmniRoute instance."""

import json
from pathlib import Path
import httpx

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
COMBOS_FILE = CONFIG_DIR / "combos.json"
OMIR_ENV = Path(__file__).resolve().parent.parent / ".env"

admin_pwd = ""
if OMIR_ENV.exists():
    with open(OMIR_ENV, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("OMNIROUTE_ADMIN_PASSWORD="):
                admin_pwd = line.split("=", 1)[1].strip()

client = httpx.Client(timeout=15.0)
client.post("http://127.0.0.1:20128/api/auth/login", json={"password": admin_pwd})

# Fetch existing combos to delete/update
existing = client.get("http://127.0.0.1:20128/api/combos").json().get("combos", [])
for c in existing:
    client.delete(f"http://127.0.0.1:20128/api/combos/{c['id']}")

with open(COMBOS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for combo in data.get("combos", []):
    payload = {
        "name": combo["name"],
        "models": combo["models"],
        "strategy": combo.get("strategy", "fill-first"),
    }
    r = client.post("http://127.0.0.1:20128/api/combos", json=payload)
    print(f"Synced {combo['name']}: Status {r.status_code}")

print("Combo synchronization complete.")
