# Harshu AI OS — V1 Local Release Candidate Evidence

## Executive Summary

This document records the definitive local verification, audit, chaos testing, and architectural evidence for **Harshu AI OS V1 (Release Candidate)**.

All evidence presented here was gathered from local execution on the candidate commit. No metrics, runtime proofs, or deployment statuses have been fabricated.

---

## 1. Verified Architecture & Request Flows

Harshu AI OS implements a single-node, truthful GenAI orchestration runtime exposing a unified FastAPI interface (`/ask`, `/ask/rag`, `/ask/agent`, `/health`, `/ready`):

1. **Direct Path (`/ask`):**
   - Direct lightweight query routing with bounded timeout, retry, and fallback policies.
   - Observability middleware generates or correlates `X-Request-ID` and outputs lifecycle JSON logs.
2. **Strict RAG Path (`/ask/rag`):**
   - Deterministic chunk hashing and incremental document ingestion.
   - Vector similarity retrieval against persistent ChromaDB SQLite storage.
   - Optional cross-encoder reranking.
   - Distance threshold gating and sufficiency judge for truthful abstention (`insufficient_context`).
   - Strict citation mapping linking claims to supporting chunk IDs.
3. **Bounded Agent ReAct Path (`/ask/agent`):**
   - Deterministic bounded step loop (`DEFAULT_MAX_STEPS = 5`).
   - Strict tool allowlist (`rag_lookup`, `web_search`) preventing arbitrary shell, network, or filesystem mutations.
   - Guardrails against repeated identical calls, malformed tool arguments, and argument bloat.
4. **Model Context Protocol (MCP) Server Adapter:**
   - Stateless server conforming to the official MCP specification (`2026-07-28`) via `mcp>=2.1.1`.
   - Read-only tools (`rag_lookup`, `system_status`) with bounds and truthful component health inspection.

---

## 2. Test Suite & Static Analysis Results

All checks executed locally from clean working environment:

- **Ruff Linter:** `uv run ruff check .` → **Clean (0 errors, 0 warnings)**.
- **Pytest Suite:** `uv run pytest` → **184 passed, 1 warning in 12.03s**.
- **Formatting / Diff Integrity:** `git diff --check` → **Clean (0 formatting conflicts)**.
- **Compose Specification:** `docker compose config` → **Syntactically and semantically valid**.

---

## 3. RAG Quality Gate Results

Executed via `uv run python src/harshu_ai_os/evaluations/run_evaluation.py`:

| Metric | Target Threshold | Measured Run 1 | Measured Run 2 | Gate Result |
| :--- | :--- | :--- | :--- | :--- |
| **Hit@3** | $\ge 25.00\%$ | **26.32%** | **26.32%** | **PASSED** |
| **MRR** | $\ge 0.1500$ | **0.1754** | **0.1754** | **PASSED** |
| **Accuracy** | $\ge 30.00\%$ | **36.84%** | **36.84%** | **PASSED** |
| **Reproducibility** | Exact Match | 100% Deterministic | 100% Deterministic | **PASSED** |

*Note: These baseline metrics validate that retrieval quality does not regress below the deterministic quality gate. They represent truthful test-corpus metrics, not claims of high retrieval accuracy.*

---

## 4. Deployment & Chaos Evidence

| Chaos / Deployment Scenario | Implementation | Evidence Status | Verification Method |
| :--- | :--- | :--- | :--- |
| **Transient LLM Flapping Recovery** | Exponential backoff retry | **RUNTIME-PROVEN** | Pytest unit/integration test with mock transient failures |
| **LLM Retry Exhaustion** | 3 bounded attempts, fail closed | **RUNTIME-PROVEN** | `test_chaos_llm_retry_exhaustion_fails_closed` (exits after 3 attempts) |
| **Permanent LLM Failure** | Immediate fail-closed, no retry | **RUNTIME-PROVEN** | `test_chaos_permanent_llm_failure_no_retry` (1 attempt only) |
| **Adversarial RAG Context** | Sufficiency judge abstention | **RUNTIME-PROVEN** | `test_chaos_adversarial_rag_empty_abstention` |
| **Oversized Request Attack** | Dual-phase 64 KiB rejection | **RUNTIME-PROVEN** | ASGI chunk streaming test returning HTTP 413 |
| **Preflight Deployment Tag Rejection** | Rejection of `latest` or empty tag | **SIMULATED** | `test_simulated_deployment_preflight_tag_rejection` & script preflight check |
| **Post-Replacement Rollback Logic** | State transition on failed health | **SIMULATED** | `test_simulated_deployment_rollback_decision_logic` |
| **Container Engine Rollback Execution** | Local Docker daemon restart | **NOT RUNTIME-PROVEN** | Local workstation Docker daemon permission denied (`/var/run/docker.sock`) |

---

## 5. Security & Observability Audit

1. **Authentication Fail-Closed:**
   - Missing `HARSHU_API_KEY` without `HARSHU_AUTH_DISABLED=true` returns HTTP 503 (service unconfigured).
   - Invalid credentials return HTTP 401 using constant-time `hmac.compare_digest`.
   - `/health` and `/ready` probes remain unauthenticated for orchestration monitors.
2. **AI Telemetry Truthfulness:**
   - Stage latencies (`retrieval_ms`, `reranking_ms`, etc.) are nullable and serialize as `null` or omitted when unexecuted, never fabricated as `0.0`.
   - Missing token usage, cost, and provider details are omitted from logs rather than fabricated as zeros or empty strings.
   - Known zeros (e.g., `tool_calls_count = 0` when zero tools were invoked) are explicitly distinguished from unknown values.
3. **Payload Sanitization & Boundary:**
   - Requests exceeding 64 KiB are rejected at the ASGI layer before JSON parsing; body contents are never logged.
   - Authorization headers, API keys, and sensitive queries are masked in structured logs.

---

## 6. Current Architectural Scaling Boundaries & Limitations

1. **Embedded ChromaDB SQLite Storage:**
   - Single-node SQLite concurrency boundary. Running multiple replicas against a single shared volume causes SQLite locking contention.
2. **In-Memory Rate Limiting:**
   - Sliding-window rate limiter is process-local and in-memory; distributed clusters require external state store (e.g., Redis).
3. **Stateless Single-Turn Memory:**
   - Conversations are not persisted across separate requests.
4. **Local Deployment Simulation:**
   - `scripts/deploy_local.sh` and `scripts/deploy_local.ps1` simulate container deployment and rollback locally. They are not replacements for Kubernetes or cloud-managed rolling deployments.
