# Harshu AI OS

Harshu AI OS is a learning-first AI engineering project with a FastAPI backend and a responsive web interface. It supports direct AI questions and retrieval-augmented generation (RAG) over locally indexed notes.

> **Learning mode:** If you are currently studying RAG evaluation metrics, do
> not read the entire repository. Start with [`LEARNING_NOW.md`](LEARNING_NOW.md),
> which limits the checkpoint to three files.

The project demonstrates model routing, embeddings, local vector retrieval, grounded answer generation, and structured citations. LangChain is used only where it adds value to the RAG prompt-model-parser workflow; the direct endpoint remains a small LiteLLM call.

## Current features

- Direct questions through `POST /ask`
- RAG questions through `POST /ask/rag`
- Automatic question-complexity classification and model routing
- Local ChromaDB vector storage
- Google GenAI embeddings
- Grounded RAG responses with source, chunk ID, chunk index, and distance citations
- Optional expandable retrieved evidence in the frontend
- Responsive desktop and mobile interface
- Friendly loading, backend, and network states
- FastAPI request and response validation with Pydantic
- Automated backend tests

## Request flow

Direct answers use the shortest path:

```text
Frontend → FastAPI → router → LiteLLM client → selected provider
```

RAG answers add retrieval and LangChain composition:

```text
Frontend → FastAPI → router → RAG service
                              ├─ Chroma retrieval
                              └─ LangChain prompt → selected model → parser
                           → answer + citations
```

## Small codebase map

```text
api/main.py
├─ /ask     → llm/router.py → llm/client.py
└─ /ask/rag → rag/service.py
               ├─ rag/chroma_store.py
               ├─ rag/embedding_client.py
               ├─ rag/ingestion.py
               └─ llm/client.py

core.py             # shared configuration and logging
```

The ordered learning references are indexed in
[`jupyter/README.md`](jupyter/README.md). They explain the harder Python,
routing, reliability, RAG, LangChain, testing, and API-flow concepts using
synthetic examples and fake model responses.

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

Use your own API keys and never commit `.env`. The repository contains only placeholder values in `.env.example`.

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

The frontend expects the backend at `http://127.0.0.1:8000` by default. To use another backend URL, set `VITE_API_BASE_URL` when starting or building the frontend.

## Ingest the synthetic example document

The repository includes a safe synthetic document at:

```text
examples/documents/harshu_ai_os_overview.txt
```

With `GEMINI_API_KEY` configured, run this command from the repository root:

```bash
uv run python scripts/ingest_documents.py
```

The script splits the example into fixed-size word chunks, generates embeddings, and upserts the chunks into the local Chroma collection. The generated database is stored under `data/chroma/` and is intentionally excluded from Git.

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

Generated frontend output is written to `frontend/dist/` and is not committed.

## Current limitations

- There is no document upload endpoint; ingestion currently uses a local script.
- Conversations and answers are not persisted between requests.
- ChromaDB storage is local to each developer environment.
- Document chunking uses a basic fixed-word strategy without semantic boundaries or overlap.
- The interface is single-turn rather than a multi-message chat.
