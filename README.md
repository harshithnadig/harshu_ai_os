# Harshu AI OS

Harshu AI OS is a learning-first AI engineering project with a FastAPI backend and a responsive web interface. It features a Unified Request Orchestrator that automatically classifies incoming questions and chooses the appropriate execution workflow (Direct LLM synthesis, multi-tool Agent, or strict document-grounded RAG).

> **Learning mode:** If you are currently studying RAG evaluation metrics or agent loops, do
> not read the entire repository at once. Start with [`LEARNING_NOW.md`](LEARNING_NOW.md),
> which gives you one small reading path at a time.

The project demonstrates request planning, model routing, local ChromaDB vector retrieval, sufficiency judging, bounded ReAct agent loops with multi-tool execution (`web_search` and `rag_lookup`), and structured citations.

## Current features (Harshu AI OS V1)

- **Unified Request Orchestration:** Single entrypoint `POST /ask` with automated complexity classification and workflow routing (`DIRECT`, `AGENT`, `STRICT_RAG`).
- **Security Baseline & Abuse Protection:**
  - Fail-closed API key authentication (`X-API-Key` or `Authorization: Bearer`) via `hmac.compare_digest`. Required by default.
  - Missing `HARSHU_API_KEY` fails closed with HTTP 503. Local development without auth requires explicit `HARSHU_AUTH_DISABLED=true`.
  - Deterministic sliding-window rate limiting (60 req/min per client IP) with HTTP 429 and `Retry-After` headers.
  - Strict 64KB request body payload limit enforced at pure ASGI middleware level (HTTP 413).
  - Question length validation capped at 4,000 characters (HTTP 422).
  - Configurable dynamic CORS allowlisting via `HARSHU_CORS_ORIGINS`.
- **Reliability & LLM Provider Hardening:**
  - Bounded exponential retries (up to 3 attempts) for transient errors (503, 429, timeouts, network drops).
  - Zero duplicate retries on permanent client/auth errors (400, 401, 403) with fail-closed domain error mapping (`LLMAuthenticationError`, `LLMTimeoutError`, `LLMRateLimitError`, `LLMServiceError`).
  - Automatic fallback route invocation when primary model is degraded.
- **AI-Specific Telemetry & Observability:**
  - Request-level correlation IDs (`X-Request-ID`) via Python `contextvars`.
  - Single-line JSON structured logs tracking workflow type, model role, complexity, stage latencies (`retrieval_ms`, `judge_ms`, `generation_ms`), tool calls count, and abstention status.
  - Strict privacy guarantees: zero raw user prompt, model response, or API secret leakage in lifecycle logs.
- **Agent & Tool Safety Guardrails:**
  - Explicit function dispatch allowlist (`AVAILABLE_TOOLS`) with bounded execution (`DEFAULT_MAX_STEPS = 5`).
  - Parameter bounds validation and consecutive repeated-call loop breaker notice.
  - Strict prohibition of arbitrary shell, OS command, or unconstrained filesystem execution.
- **Model Context Protocol (MCP) v1:**
  - Official Model Context Protocol SDK v2 integration conforming to the 2026-07-28 specification.
  - Stateless discovery via `server/discover` and tool execution via `tools/call`.
  - Exposes strictly validated, safe read-only capabilities (`rag_lookup`, `system_status`) with legacy `initialize` backward-compatibility.
- **Health & Readiness Architecture:**
  - Liveness probe `GET /health`: ultra-fast, zero-dependency process heartbeat (HTTP 200).
  - Readiness probe `GET /ready`: dependency verification checking local ChromaDB vector store health (HTTP 200 or 503).
- **Deployment & Automated Rollback:**
  - Cross-platform local production simulation (`scripts/deploy_local.sh` and `scripts/deploy_local.ps1`).
  - Tag vs cryptographically pinned image digest verification.
  - Automated health polling and immediate container rollback on deployment failures.

## Request flow

All user questions are sent to the primary entrypoint `POST /ask`:

```text
User → POST /ask
        │
        ▼
Request Planner (classify complexity, information source & grounding requirement)
        │
        ▼
Validated RequestPlan
        │
        ├─ DIRECT      ──► LiteLLM client ──► Selected Model Route
        │
        ├─ AGENT       ──► Bounded ReAct Loop (tools: web_search, rag_lookup)
        │
        └─ STRICT_RAG  ──► Chroma retrieval ──► Distance gate ──► Sufficiency Judge ──► Grounded Synthesis
```

### Workflow Execution Details

1. **DIRECT:** Used for ordinary questions that do not require external web retrieval or internal project documents.
2. **AGENT:** Used for requests requiring dynamic tool execution, live external information (`web_search`), internal knowledge retrieval (`rag_lookup`), or mixed requirements.
3. **STRICT_RAG:** Used when questions explicitly require strict grounding against indexed project documents, returning verified citations and abstaining if evidence is insufficient.

> **Note on Diagnostic Endpoints:** `POST /ask/rag` and `POST /ask/agent` remain available as diagnostic and development endpoints for isolated testing, but are not required for normal frontend usage.

## Small codebase map

```text
api/main.py
├─ /ask       → orchestrator/service.py
│                ├─ llm/router.py (RequestPlan, choose_route)
│                ├─ llm/client.py (direct call)
│                ├─ agents/loop.py (bounded ReAct agent loop)
│                └─ rag/service.py (strict RAG pipeline)
├─ /ask/agent → agents/loop.py (diagnostic endpoint)
└─ /ask/rag   → rag/service.py (diagnostic endpoint)
                 ├─ rag/chroma_store.py
                 ├─ rag/embedding_client.py
                 └─ rag/judge.py

observability/  # request IDs, contextvars, pure ASGI middleware, structured JSON logging
core.py         # shared configuration and logging
```

The ordered learning references are indexed in [`jupyter/README.md`](jupyter/README.md).

## Prerequisites

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js with npm

## Environment variables

Copy the provided example file to a local `.env` file:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Set the required provider credentials inside `.env`:

```dotenv
HARSHU_AI_OS_MODE=development
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Use your own API keys and never commit `.env`.

## Backend setup

From the repository root, install the Python dependencies:

```bash
uv sync
```

Start the FastAPI development server:

```bash
uv run uvicorn harshu_ai_os.api.main:app --reload
```

The backend runs at `http://127.0.0.1:8000`.

Interactive FastAPI documentation is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Running with Docker and Docker Compose

You can run Harshu AI OS locally using standalone Docker or multi-service orchestration with Docker Compose.

### Option A: Docker Compose (Recommended)

Docker Compose provides a local production-style environment with automatic volume persistence, internal network isolation, container health checks, and declarative environment variable management.

#### 1. Start the services
Start the container in the background (detached mode) with build on launch:

```bash
docker compose up -d --build
```

Or run interactively in the foreground to stream logs directly to your terminal:

```bash
docker compose up
```

#### 2. Check service status
Inspect the state and health of running Compose services:

```bash
docker compose ps
```

View live real-time service logs:

```bash
docker compose logs -f app
```

#### 3. Access API and Health endpoint
Verify the FastAPI backend is healthy:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy"}
```

Access the interactive API docs at `http://localhost:8000/docs`.

#### 4. Stop the services
Stop and remove containers and internal networks while preserving persistent volumes:

```bash
docker compose down
```

To remove containers and volumes completely:

```bash
docker compose down -v
```

---

### Option B: Standalone Docker

#### 1. Build the Docker image

```bash
docker build -t harshu-ai-os .
```

#### 2. Run the container

Run the container using your local `.env` file for configuration:

```bash
docker run -p 8000:8000 -v "${PWD}/data:/app/data" --env-file .env harshu-ai-os
```

Or pass individual environment variables:

```bash
docker run -p 8000:8000 -v "${PWD}/data:/app/data" \
  -e HARSHU_AI_OS_MODE=development \
  -e GEMINI_API_KEY=your_gemini_key \
  -e GROQ_API_KEY=your_groq_key \
  harshu-ai-os
```

#### 3. Verify the container

Test the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy"}
```

---

### Container Architecture & Persistence

- **Embedded ChromaDB:** Chroma runs directly inside the FastAPI application process using `chromadb.PersistentClient(path="data/chroma")`. There is no separate Chroma database service required, which minimizes memory overhead and network latency for single-instance deployments.
- **Data Persistence:** The host directory `./data` is mounted to `/app/data` inside the container. This guarantees that vector embeddings, chunk collections, and SQLite indexes created during document ingestion survive container restarts and rebuilds.
- **Networking & DNS:** Docker Compose establishes an isolated user-defined bridge network (`harshu-network`). Containers on this network can communicate using their service names as hostnames (e.g. `http://app:8000`). Within containers, `localhost` refers only to that individual container's network namespace; inter-container communication relies on Compose service names resolved by Docker's built-in DNS.

---

## Container Image Registry (GHCR)

The Harshu AI OS container image is automatically built, tested, and published to the **GitHub Container Registry (GHCR)** on every push to `main` via the automated CI/CD workflow.

### Registry Location
```text
ghcr.io/harshithnadig/harshu_ai_os
```

### Pulling the Image

To pull the latest release image:

```bash
docker pull ghcr.io/harshithnadig/harshu_ai_os:latest
```

To pull a specific commit-pinned build (recommended for deterministic release candidate deployments and rollbacks):

```bash
docker pull ghcr.io/harshithnadig/harshu_ai_os:sha-<commit_sha>
```

### Running the Pulled Image

Run the container using your local persistent data volume and environment file:

```bash
docker run -d \
  --name harshu-ai-os \
  -p 8000:8000 \
  -v "${PWD}/data:/app/data" \
  --env-file .env \
  ghcr.io/harshithnadig/harshu_ai_os:latest
```

### Tagging & Rollback Strategy

| Tag Format | Example | Mutability | Intended Use Case |
| :--- | :--- | :--- | :--- |
| `latest` | `ghcr.io/harshithnadig/harshu_ai_os:latest` | Mutable | Local development and default pulling of the latest `main` build. |
| `<branch>` | `ghcr.io/harshithnadig/harshu_ai_os:main` | Mutable | Tracking the head of a specific branch. |
| `sha-<commit_sha>` | `ghcr.io/harshithnadig/harshu_ai_os:sha-dde3db1` | Mutable Tag (CI-pinned) | Release & rollback target convention. Pinned by CI to the git commit SHA. |
| `@sha256:<digest>` | `ghcr.io/harshithnadig/harshu_ai_os@sha256:...` | **Immutable** | **Content-addressed immutable identity.** Cryptographically pinned artifact. |

### Local Deployment Simulation

Harshu AI OS includes an automated local deployment script (`scripts/deploy_local.ps1`) to simulate production-style zero-downtime-conscious release workflows on a local workstation without requiring paid cloud infrastructure.

- **Why SHA-Tagged Deployments:** Deployments strictly reject `latest` and require specific commit tags (e.g. `sha-3d4f8eb`) to track releases back to source commits. Note that while tags are conventional pointers that could technically be reassigned in a registry, the truly content-addressed immutable artifact identifier is the SHA256 image digest (`ghcr.io/...@sha256:<digest>`).
- **Deploy Command:**
  ```powershell
  .\scripts\deploy_local.ps1 -ImageTag sha-3d4f8eb
  ```
- **Health Verification:** After container instantiation, the deployment script executes bounded polling against `http://localhost:8000/health` requiring HTTP 200 and `{"status":"healthy"}` before marking deployment as successful.
- **Automatic Rollback:** If the candidate image fails to start or fails the health check within the timeout window, the script automatically terminates the candidate, restores the previously running container version with identical environment and volume bindings, and verifies health of the restored service.
- **Scope & Limitations:** This is a **local production simulation** running a single standalone container bound to local host ports and `./data` storage. It does not replace cloud-native production architectures (such as Kubernetes orchestrators, multi-replica horizontal autoscaling, external managed vector databases, or global load balancers).

---

## Observability (v1)

Harshu AI OS provides request-level observability to trace every HTTP request through structured logs with low overhead.

```text
HTTP Request
  ↓
Request ID (X-Request-ID Header or Generated UUID)
  ↓
Concurrency-Safe Context (Python contextvars)
  ↓
Structured JSON Logs (http_request_started, http_request_completed, http_request_failed)
  ↓
Status + Latency (time.perf_counter) + Error Correlation
```

### Key Features
- **Request ID Propagation:** Client-supplied `X-Request-ID` headers are validated and sanitized (or a fresh UUID4 is generated), stored in `contextvars`, and returned in the `X-Request-ID` response header.
- **Structured JSON Logging:** Emits machine-readable single-line JSON log entries for all request lifecycle events and application logs.
- **Latency & Error Tracking:** Server-side request duration (`duration_ms`) is measured via `time.perf_counter()`. Unhandled exceptions emit `http_request_failed` with safe error type classification without exposing sensitive payloads or raw exception strings.
- **Scope Notice:** This milestone implements **Observability v1** (request correlation, latency, and structured logging). It does not include distributed tracing (Jaeger/OpenTelemetry), Prometheus metrics exporters, Grafana dashboards, or external monitoring SaaS.

---

## Frontend setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend at `http://localhost:5173`.

## Ingest the synthetic example document

With `GEMINI_API_KEY` configured, run this command from the repository root:

```bash
uv run python scripts/ingest_documents.py
```

The script splits the example into fixed-size chunks, generates embeddings, and upserts the chunks into the local Chroma collection under `data/chroma/`.

## Verification

Run the complete backend test suite from the repository root:

```bash
uv run pytest
```

Run frontend checks from the `frontend` directory:

```bash
npm run lint
npm run build
```

## Security Baseline & Abuse Protection

Harshu AI OS enforces defense-in-depth protections across the request lifecycle:

1. **Fail-Closed API Key Authentication (`api/security.py`):**
   - **Authentication is required by default.** Protected endpoints (`POST /ask`, `POST /ask/rag`, `POST /ask/agent`) strictly enforce credentials. Public endpoints (`GET /health`, `GET /ready`) remain accessible without credentials.
   - Supports `X-API-Key: <key>` and `Authorization: Bearer <key>`.
   - Uses constant-time string comparison (`hmac.compare_digest`) to resist timing side-channel attacks.
   - **Fail-Closed Guarantee:** If `HARSHU_API_KEY` is not configured, protected endpoints fail closed with HTTP 503 Service Unavailable (`Authentication service unavailable: server authentication is unconfigured`) without exposing server internals or secrets.
   - **Explicit Local Development Bypass:** Unauthenticated requests are permitted *only* when `HARSHU_AUTH_DISABLED=true` (or `1`, `yes`) is explicitly configured in the local environment. Never enable this flag in a deployed, staging, or shared environment.
2. **Deterministic Sliding-Window Rate Limiter:**
   - Thread-safe in-memory rate limiter tracking client IP addresses across a rolling 60-second window.
   - Default threshold: 60 requests/minute. Rejections return HTTP 429 with standard `Retry-After: <seconds>` header.
3. **Payload & Input Size Constraints:**
   - Pure ASGI middleware check rejecting any incoming request payload exceeding 64 KB (65,536 bytes) with HTTP 413 before parsing JSON or allocating memory.
   - Pydantic schema validation restricting question inputs to `max_length=4000` (HTTP 422).
4. **CORS Control:**
   - Configurable via `HARSHU_CORS_ORIGINS` (comma-delimited), defaulting to local web dev ports (`localhost:5173`, `127.0.0.1:5173`).

---

## Model Context Protocol (MCP) v1

Harshu AI OS includes a safe, read-only Model Context Protocol (MCP) server adapter conforming to the current official specification (`2026-07-28`), powered by the official Model Context Protocol Python SDK v2 (`mcp>=2.1.1`):

- **Architecture:** Modern stateless protocol core removing the legacy session requirement.
- **Protocol Discovery & Execution:**
  - Modern clients discover server capabilities, identity, and supported versions via `server/discover`.
  - Tools are discovered via `tools/list` and executed via `tools/call`.
  - Backwards-compatibility: Legacy clients utilizing the older `initialize` handshake are automatically supported via the official SDK compatibility layer.
- **Exposed Read-Only Capabilities:**
  - `rag_lookup`: Search indexed local knowledge base. Enforces strict bounds on input types, query length (capped at 1,000 characters), and result limits ($1 \le k \le 10$).
  - `system_status`: Inspect truthful, live component health (including actual vector store probe) and runtime capabilities without static unverified assertions.
- **Safety Guarantee:** MCP integration is strictly read-only. Unlisted tools (shell execution, arbitrary file writes, destructive operations) are safely rejected with `is_error: true`.

---

## Architectural Decision Records (ADRs)

Key architectural boundaries are documented in [`docs/architecture_decisions.md`](docs/architecture_decisions.md):
- **ADR-001 (LangGraph Evaluation):** Deferred for V1. Pure Python bounded ReAct loops (`DEFAULT_MAX_STEPS = 5`) fulfill all current agent requirements without the graph complexity, checkpointer overhead, or latency of LangGraph.
- **ADR-002 (Persistence & Queues):** Deferred for V1. Local persistent ChromaDB and thread-safe in-memory rate limiting satisfy single-node requirements with zero external container dependencies. External databases (Postgres/Redis/Celery) will be evaluated when distributed horizontal scaling is required.

---

## Current limitations & Scaling Boundaries

1. **Embedded ChromaDB Scaling Boundary:**
   - Harshu AI OS uses local persistent ChromaDB (`chromadb.PersistentClient`) with SQLite storage. SQLite has a single-writer lock; running multiple container replicas sharing the same volume will encounter SQLite locking contention.
   - For multi-replica horizontal autoscaling in Kubernetes/Cloud Run, an external client/server vector database (such as standalone Chroma server, Qdrant, or Pinecone) is required.
2. **Single-Node Rate Limiting:**
   - The sliding-window rate limiter is currently in-memory. In a distributed multi-node cluster, a centralized cache (e.g., Redis) would be needed to enforce global rate limits across all nodes.
3. **Single-Turn Memory Model:**
   - The runtime currently handles stateless single-turn requests; multi-message session history is not persisted between requests.
4. **Local Deployment Simulation vs Cloud Production:**
   - `scripts/deploy_local.sh` and `scripts/deploy_local.ps1` simulate production deployment verification and rollbacks on a local single-node workstation. They do not replace managed cloud ingress, multi-region load balancers, or Kubernetes rolling update controllers.

- There is no document upload endpoint; ingestion currently uses a local script.
- Conversations and answers are not persisted between requests.
- ChromaDB storage is local to each developer environment.
- Document chunking uses a basic fixed-word strategy without semantic boundaries or overlap.
- The interface is single-turn rather than a multi-message chat.
