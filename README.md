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
