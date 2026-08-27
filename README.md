# Harshu AI OS

Harshu AI OS is a learning-first AI engineering project with a FastAPI backend and a responsive web interface. It features a Unified Request Orchestrator that automatically classifies incoming questions and chooses the appropriate execution workflow (Direct LLM synthesis, multi-tool Agent, or strict document-grounded RAG).

> **Learning mode:** If you are currently studying RAG evaluation metrics or agent loops, do
> not read the entire repository at once. Start with [`LEARNING_NOW.md`](LEARNING_NOW.md),
> which gives you one small reading path at a time.

The project demonstrates request planning, model routing, local ChromaDB vector retrieval, sufficiency judging, bounded ReAct agent loops with multi-tool execution (`web_search` and `rag_lookup`), and structured citations.

## Current features

- Unified question endpoint through `POST /ask` with automated planning and workflow dispatch
- Diagnostic development endpoints `POST /ask/rag` and `POST /ask/agent`
- Automatic question complexity and information requirement classification via `RequestPlan`
- Deterministic workflow selection:
  - **DIRECT:** Fast language model synthesis for ordinary knowledge questions
  - **AGENT:** Bounded ReAct agent loop supporting `web_search` and `rag_lookup` with deterministic multi-domain coverage guards
  - **STRICT_RAG:** Grounded RAG with ChromaDB retrieval, cosine distance gating, LLM sufficiency judge, supporting chunks, and citation/abstention guarantees
- Local ChromaDB vector storage and embeddings routed through the OmniRoute logical embedding role
- Responsive desktop and mobile interface with telemetry inspector drawer
- Friendly loading, backend, and network states
- FastAPI request and response validation with Pydantic
- Automated backend and unit test suites

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

To pull a specific immutable commit build (recommended for deterministic production deployments and rollbacks):

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
| `sha-<commit_sha>` | `ghcr.io/harshithnadig/harshu_ai_os:sha-dde3db1` | **Immutable** | **Production deployments & rollback target.** Pinned directly to the git commit SHA. |

### Local Deployment Simulation

Harshu AI OS includes an automated local deployment script (`scripts/deploy_local.ps1`) to simulate production-style zero-downtime-conscious release workflows on a local workstation without requiring paid cloud infrastructure.

- **Why Immutable SHA Tags:** Deployments strictly reject `latest` and require immutable commit tags (e.g. `sha-3d4f8eb`). This guarantees that the exact binary artifact tested in CI is deployed without risk of tag drift or cache pollution.
- **Deploy Command:**
  ```powershell
  .\scripts\deploy_local.ps1 -ImageTag sha-3d4f8eb
  ```
- **Health Verification:** After container instantiation, the deployment script executes bounded polling against `http://localhost:8000/health` requiring HTTP 200 and `{"status":"healthy"}` before marking deployment as successful.
- **Automatic Rollback:** If the candidate image fails to start or fails the health check within the timeout window, the script automatically terminates the candidate, restores the previously running container version with identical environment and volume bindings, and verifies health of the restored service.
- **Scope & Limitations:** This is a **local production simulation** running a single standalone container bound to local host ports and `./data` storage. It does not replace cloud-native production architectures (such as Kubernetes orchestrators, multi-replica horizontal autoscaling, external managed vector databases, or global load balancers).

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

## Current limitations

- There is no document upload endpoint; ingestion currently uses a local script.
- Conversations and answers are not persisted between requests.
- ChromaDB storage is local to each developer environment.
- Document chunking uses a basic fixed-word strategy without semantic boundaries or overlap.
- The interface is single-turn rather than a multi-message chat.
