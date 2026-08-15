"""Secure credential rotation script for OmniRoute.
Rotates admin password and API keys without printing secrets to stdout.
"""

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
import httpx

SUB_DIR = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = SUB_DIR / "upstream" / "OmniRoute"
OMIR_ENV = SUB_DIR / ".env"

# 1. Generate secure high-entropy admin password
new_admin_password = secrets.token_hex(24)

# Reset admin password in sqlite via OmniRoute CLI
proc = subprocess.Popen(
    ["node", "bin/reset-password.mjs", "--password-stdin"],
    cwd=str(UPSTREAM_DIR),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
stdout, stderr = proc.communicate(input=new_admin_password)
if proc.returncode != 0:
    print(f"Password reset failed: {stderr}")
    sys.exit(1)

print("[SECURITY] Admin password successfully reset to secure random credential.")

# 2. Log in with the new password
client = httpx.Client(timeout=15.0)
login_res = client.post("http://127.0.0.1:20128/api/auth/login", json={"password": new_admin_password})
if login_res.status_code != 200:
    print(f"[SECURITY] Login with new password failed: HTTP {login_res.status_code}")
    sys.exit(1)

# 3. List and revoke existing API keys
keys_res = client.get("http://127.0.0.1:20128/api/keys")
existing_keys = keys_res.json().get("keys", [])
revoked_count = 0
for k in existing_keys:
    client.delete(f"http://127.0.0.1:20128/api/keys/{k['id']}")
    revoked_count += 1

print(f"[SECURITY] Revoked {revoked_count} old API key(s).")

# 4. Create new replacement API key
create_res = client.post(
    "http://127.0.0.1:20128/api/keys",
    json={"name": "harshu_gateway_secure_key"},
)
if create_res.status_code != 201:
    print(f"[SECURITY] Key creation failed: HTTP {create_res.status_code}")
    sys.exit(1)

new_key_data = create_res.json()
new_token = new_key_data.get("key") or new_key_data.get("apiKey") or new_key_data.get("token")

# 5. Write credentials to omiroute/.env securely
with open(OMIR_ENV, "w", encoding="utf-8") as f:
    f.write(f"OMNIROUTE_ADMIN_PASSWORD={new_admin_password}\n")
    f.write(f"OMNIROUTE_API_KEY={new_token}\n")

print("[SECURITY] New replacement API key successfully created and written to local omiroute/.env.")
print("[SECURITY] Credential rotation completed successfully. Zero secrets displayed.")
