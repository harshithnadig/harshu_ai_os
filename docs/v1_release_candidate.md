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
   - Guardrails against repeated identical calls, malformed tool arguments, and string argument truncation at 2,000 characters (`src/harshu_ai_os/agents/loop.py:104-105`).
4. **Model Context Protocol (MCP) Server Adapter:**
   - Stateless server conforming to the official MCP specification (`2026-07-28`) via `mcp>=2.1.1`.
   - Read-only tools: `rag_lookup` (query bounded to 1,000 characters in `src/harshu_ai_os/mcp/server.py:62-63`) and `system_status` with truthful component health inspection.

---

## 2. Test Suite & Static Analysis Results

All checks executed locally from clean working environment:

- **Ruff Linter:** `uv run ruff check .` → **Clean (0 errors, 0 warnings)**.
- **Pytest Suite:** `uv run pytest` → **184 passed, 1 warning in 12.01s**.
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

Strict Evidence Classification:
- **TESTED:** Deterministic test exercised local code behavior.
- **SIMULATED:** Failure or environment condition was injected or modeled.
- **RUNTIME-PROVEN:** Actual relevant runtime, component, or script path was executed.
- **HISTORICALLY RUNTIME-PROVEN:** Validated by prior real execution in repository history/commits.
- **NOT RUNTIME PROVEN:** Not executed against the live subsystem in the current session.

| Chaos / Deployment Scenario | Implementation | Evidence Status | Verification Method |
| :--- | :--- | :--- | :--- |
| **Transient LLM Flapping Recovery** | Exponential backoff retry | **SIMULATED / TESTED** | `test_1_primary_service_unavailable_retries_then_activates_fallback` (injected transient 503) |
| **LLM Retry Exhaustion** | 3 bounded attempts, fail closed | **SIMULATED / TESTED** | `test_chaos_llm_retry_exhaustion_fails_closed` (injected persistent 503, fails closed after 3 attempts) |
| **Permanent LLM Failure** | Immediate fail-closed, no retry | **SIMULATED / TESTED** | `test_chaos_permanent_llm_failure_no_retry` (injected auth error, fails immediately on attempt 1) |
| **Adversarial RAG Context** | Sufficiency judge abstention | **TESTED** | `test_chaos_adversarial_rag_empty_abstention` (real RAG components with synthetic empty/unrelated input) |
| **Oversized Request Attack** | Dual-phase 64 KiB rejection | **TESTED** | `test_payload_size_limit_rejected` & ASGI chunk streaming tests returning HTTP 413 |
| **Healthy Local Deployment** | Container start & health polling | **HISTORICALLY RUNTIME-PROVEN** | Proven in commit `c5948ef`; SIMULATED in current session |
| **Bad / Missing Image Preflight** | Rejection of `latest` or bad tag | **HISTORICALLY RUNTIME-PROVEN** / **TESTED (Session)** | Tested live in commit `c5948ef`; Python preflight logic tested via `test_simulated_deployment_preflight_tag_rejection` |
| **Post-Replacement Rollback Logic** | State transition on failed health | **SIMULATED** | `test_simulated_deployment_rollback_decision_logic` |
| **Container Engine Rollback Execution** | Local Docker daemon restart | **NOT RUNTIME PROVEN** | Local workstation Docker daemon permission denied (`/var/run/docker.sock`) |

---

## 5. Security & Observability Audit

1. **Authentication Fail-Closed:**
   - Missing `HARSHU_API_KEY` without `HARSHU_AUTH_DISABLED=true` returns HTTP 503 (service unconfigured).
   - Invalid credentials return HTTP 401 using constant-time `hmac.compare_digest`.
   - `/health` and `/ready` probes remain unauthenticated for orchestration monitors.
2. **Request ID Validation & Bounds:**
   - Client-supplied `X-Request-ID` headers are sanitized and bounded to 128 characters (`MAX_REQUEST_ID_LENGTH = 128` in `src/harshu_ai_os/observability/context.py:11`). Overly long or malformed IDs are rejected and replaced with a valid UUID4 (`test_malformed_and_overly_long_request_ids_safely_handled`).
3. **AI Telemetry Truthfulness:**
   - Stage latencies (`retrieval_ms`, `reranking_ms`, etc.) are nullable and serialize as `null` or omitted when unexecuted, never fabricated as `0.0`.
   - Missing token usage, cost, and provider details are omitted from logs rather than fabricated as zeros or empty strings.
   - Known zeros (e.g., `tool_calls_count = 0` when zero tools were invoked) are explicitly distinguished from unknown values.
4. **Payload Sanitization & Boundary:**
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
