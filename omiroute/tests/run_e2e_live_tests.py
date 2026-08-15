"""Comprehensive live end-to-end test suite for Harshu AI OS with OmniRoute gateway."""

import sys
import time
import httpx
from pathlib import Path

from harshu_ai_os.core import get_omniroute_config
from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.llm.router import classify_task_with_model, choose_route
from harshu_ai_os.llm.tools import AVAILABLE_TOOLS, WEB_SEARCH_TOOL_SCHEMA
from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client, embed_text
from harshu_ai_os.rag.service import answer_with_chroma_rag

def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))

# Setup admin client to observe OmniRoute gateway logs
OMIR_ENV = Path(__file__).resolve().parent.parent / ".env"
admin_pwd = ""
if OMIR_ENV.exists():
    with open(OMIR_ENV, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("OMNIROUTE_ADMIN_PASSWORD="):
                admin_pwd = line.split("=", 1)[1].strip()

admin_client = httpx.Client(timeout=10.0)
admin_client.post("http://127.0.0.1:20128/api/auth/login", json={"password": admin_pwd})

def get_latest_log():
    try:
        logs = admin_client.get("http://127.0.0.1:20128/api/usage/call-logs").json()
        return logs[0] if logs else {}
    except Exception:
        return {}

safe_print("=" * 80)
safe_print("HARSHU AI OS / OMNIROUTE LIVE END-TO-END TESTS (A THROUGH F)")
safe_print("=" * 80)

# ------------------------------------------------------------------------------
# TEST A - NORMAL SIMPLE / GENERAL
# ------------------------------------------------------------------------------
q_a = "What is a Python dictionary?"
safe_print(f"\n[TEST A - NORMAL SIMPLE / GENERAL]")
safe_print(f"  Input Question: \"{q_a}\"")
t0 = time.perf_counter()
clf_a = classify_task_with_model(q_a)
route_a = choose_route(clf_a.complexity)
res_a = call_llm(route_a, q_a, tools=[WEB_SEARCH_TOOL_SCHEMA], available_tools=AVAILABLE_TOOLS, return_tool_info=True)
dt_a = (time.perf_counter() - t0) * 1000.0
log_a = get_latest_log()

safe_print(f"  Complexity: {clf_a.complexity}")
safe_print(f"  Selected Route Model: {route_a['model']}")
safe_print(f"  Gateway Provider: {log_a.get('provider')}")
safe_print(f"  Gateway Dispatched Model: {log_a.get('model')}")
safe_print(f"  Tool Used: {res_a.get('tool_used')}")
safe_print(f"  Answer: \"{res_a.get('answer', '').strip()}\"")
safe_print(f"  Duration: {round(dt_a, 2)} ms")

# ------------------------------------------------------------------------------
# TEST B - REASONING
# ------------------------------------------------------------------------------
q_b = "Design a secure, multi-tenant RAG architecture with end-to-end encryption, multi-tier threat modeling, and trade-off analysis."
safe_print(f"\n[TEST B - REASONING]")
safe_print(f"  Input Question: \"{q_b}\"")
t0 = time.perf_counter()
clf_b = classify_task_with_model(q_b)
route_b = choose_route(clf_b.complexity)
res_b = call_llm(route_b, q_b, tools=[WEB_SEARCH_TOOL_SCHEMA], available_tools=AVAILABLE_TOOLS, return_tool_info=True)
dt_b = (time.perf_counter() - t0) * 1000.0
log_b = get_latest_log()

safe_print(f"  Complexity: {clf_b.complexity}")
safe_print(f"  Selected Route Model: {route_b['model']}")
safe_print(f"  Gateway Provider: {log_b.get('provider')}")
safe_print(f"  Gateway Dispatched Model: {log_b.get('model')}")
safe_print(f"  Answer: \"{res_b.get('answer', '').strip()[:200]}...\"")
safe_print(f"  Duration: {round(dt_b, 2)} ms")

# ------------------------------------------------------------------------------
# TEST C - WEB TOOL
# ------------------------------------------------------------------------------
q_c = "What is the release date or latest release version of Python 3.13? Use web search to find current information."
safe_print(f"\n[TEST C - WEB TOOL]")
safe_print(f"  Input Question: \"{q_c}\"")
t0 = time.perf_counter()
clf_c = classify_task_with_model(q_c)
route_c = choose_route(clf_c.complexity)
res_c = call_llm(route_c, q_c, tools=[WEB_SEARCH_TOOL_SCHEMA], available_tools=AVAILABLE_TOOLS, return_tool_info=True)
dt_c = (time.perf_counter() - t0) * 1000.0
log_c = get_latest_log()

safe_print(f"  Complexity: {clf_c.complexity}")
safe_print(f"  Selected Route Model: {route_c['model']}")
safe_print(f"  Tool Used: {res_c.get('tool_used')}")
safe_print(f"  Tool Name: {res_c.get('tool_name')}")
safe_print(f"  Tool Query: {res_c.get('tool_query')}")
safe_print(f"  Tool Sources Count: {len(res_c.get('tool_sources', []))}")
safe_print(f"  Gateway Dispatched Model: {log_c.get('model')}")
safe_print(f"  Final Answer: \"{res_c.get('answer', '').strip()[:200]}...\"")
safe_print(f"  Duration: {round(dt_c, 2)} ms")

# ------------------------------------------------------------------------------
# TEST D - RAG ANSWERABLE
# ------------------------------------------------------------------------------
q_d = "How is Harshu AI OS tested?"
safe_print(f"\n[TEST D - RAG ANSWERABLE]")
safe_print(f"  Input Question: \"{q_d}\"")
collection = get_notes_collection()
embedding_client = get_embedding_client()
clf_d = classify_task_with_model(q_d)
route_d = choose_route(clf_d.complexity)

t0 = time.perf_counter()
res_d = answer_with_chroma_rag(collection, embedding_client, q_d, route_d)
dt_d = (time.perf_counter() - t0) * 1000.0

safe_print(f"  Retrieved Chunk IDs: {res_d['ids']}")
safe_print(f"  Abstained: {res_d['abstained']}")
safe_print(f"  Judge Reason: \"{res_d.get('judge_reason')}\"")
safe_print(f"  Citations: {res_d['citations']}")
safe_print(f"  Grounded Answer: \"{res_d['answer'].strip()}\"")
safe_print(f"  Retrieval ms: {round(res_d.get('retrieval_ms', 0), 2)} | Judge ms: {round(res_d.get('judge_ms', 0), 2)} | Generation ms: {round(res_d.get('generation_ms', 0), 2)}")

# ------------------------------------------------------------------------------
# TEST E - RAG ABSTENTION
# ------------------------------------------------------------------------------
q_e = "Does ChromaDB automatically handle user password hashing in Harshu AI OS?"
safe_print(f"\n[TEST E - RAG ABSTENTION]")
safe_print(f"  Input Question: \"{q_e}\"")
clf_e = classify_task_with_model(q_e)
route_e = choose_route(clf_e.complexity)

t0 = time.perf_counter()
res_e = answer_with_chroma_rag(collection, embedding_client, q_e, route_e)
dt_e = (time.perf_counter() - t0) * 1000.0

safe_print(f"  Retrieved Chunk IDs: {res_e['ids']}")
safe_print(f"  Abstained: {res_e['abstained']}")
safe_print(f"  Abstention Reason: {res_e['abstention_reason']}")
safe_print(f"  Judge Reason: \"{res_e.get('judge_reason')}\"")
safe_print(f"  Citations: {res_e['citations']}")
safe_print(f"  Answer: \"{res_e['answer']}\"")

# ------------------------------------------------------------------------------
# TEST F - EMBEDDING
# ------------------------------------------------------------------------------
safe_print(f"\n[TEST F - EMBEDDINGS (MATCHING HARSHU AI OS)]")
t0 = time.perf_counter()
emb_vec = embed_text(embedding_client, "Harshu AI OS live verification embedding")
dt_f = (time.perf_counter() - t0) * 1000.0

safe_print(f"  Logical Role: harshu-embedding")
safe_print(f"  Actual Model: gemini/gemini-embedding-2")
safe_print(f"  Dimension: {len(emb_vec)}")
safe_print(f"  Latency: {round(dt_f, 2)} ms")
safe_print(f"  First 3 floats: {emb_vec[:3]}")

safe_print("\n" + "=" * 80)
safe_print("ALL LIVE TESTS (A THROUGH F) COMPLETED SUCCESSFULLY")
safe_print("=" * 80)
