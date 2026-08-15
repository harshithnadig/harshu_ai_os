"""Probe all candidate model IDs on Groq and Gemini for:
1. Basic chat completion
2. Structured JSON response
3. Tool / function calling
4. Latency
"""

import json
import time
from pathlib import Path
import httpx

SUB_DIR = Path(__file__).resolve().parent.parent
OMIR_ENV = SUB_DIR / ".env"

token = ""
if OMIR_ENV.exists():
    with open(OMIR_ENV, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("OMNIROUTE_API_KEY="):
                token = line.split("=", 1)[1].strip()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

candidates = [
    # Groq candidates
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
    "groq/qwen/qwen3-32b",
    # Gemini candidates
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-flash-latest",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
]

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

print("=" * 70)
print("PROBING CANDIDATE MODELS")
print("=" * 70)

for model in candidates:
    print(f"\n[MODEL] {model}")
    
    # 1. Basic Chat Test
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Respond with 'READY' in one word."}],
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        r_chat = httpx.post("http://127.0.0.1:20128/v1/chat/completions", headers=headers, json=chat_payload, timeout=12.0)
        dt_chat = (time.perf_counter() - t0) * 1000.0
        chat_ok = r_chat.status_code == 200
        chat_resp = r_chat.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip() if chat_ok else r_chat.text[:100]
        print(f"  Chat: HTTP {r_chat.status_code} ({round(dt_chat, 1)}ms) -> {chat_resp}")
    except Exception as e:
        print(f"  Chat: ERROR -> {e}")
        continue

    if not chat_ok:
        continue

    # 2. JSON Test
    json_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON: {\"status\": \"ok\", \"count\": 42}"},
            {"role": "user", "content": "Generate JSON"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        r_json = httpx.post("http://127.0.0.1:20128/v1/chat/completions", headers=headers, json=json_payload, timeout=12.0)
        json_ok = r_json.status_code == 200
        raw_j = r_json.json().get("choices", [{}])[0].get("message", {}).get("content", "") if json_ok else ""
        parsed = json.loads(raw_j) if json_ok else {}
        print(f"  JSON: HTTP {r_json.status_code} -> Valid JSON: {isinstance(parsed, dict) and 'status' in parsed}")
    except Exception as e:
        print(f"  JSON: ERROR -> {e}")

    # 3. Tool Call Test
    tool_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo? Use the tool."}],
        "tools": tools_schema,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    try:
        r_tool = httpx.post("http://127.0.0.1:20128/v1/chat/completions", headers=headers, json=tool_payload, timeout=12.0)
        tool_ok = r_tool.status_code == 200
        msg = r_tool.json().get("choices", [{}])[0].get("message", {}) if tool_ok else {}
        tcs = msg.get("tool_calls", [])
        print(f"  Tools: HTTP {r_tool.status_code} -> Tool calls emitted: {len(tcs)} ({tcs[0]['function']['name'] if tcs else 'None'})")
    except Exception as e:
        print(f"  Tools: ERROR -> {e}")
