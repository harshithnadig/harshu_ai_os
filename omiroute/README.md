# Harshu AI OS — `omiroute` Gateway Subsystem

`omiroute` is the dedicated, isolated Model/Provider Gateway subsystem for **Harshu AI OS**, embedding the complete, full-featured upstream [OmniRoute](https://github.com/diegosouzapw/OmniRoute) distribution.

---

## 1. Why This Subsystem Exists

During development of Harshu AI OS, direct provider API calls revealed a fundamental reliability bottleneck:
- When a free-tier quota is reached on a single provider (such as a Google Gemini `429 RESOURCE_EXHAUSTED` error), the entire application request would fail.
- Hardcoding provider names and model IDs throughout application logic couples business workflows (such as question routing, RAG retrieval, sufficiency judging, and web search) directly to external vendor uptime.

`omiroute` solves this by introducing a clean separation of concerns:
> **Harshu AI OS decides WHAT work needs to happen.**
> **`omiroute` decides WHICH compatible model/provider executes that work.**

---

## 2. What OmniRoute Is

[OmniRoute](https://github.com/diegosouzapw/OmniRoute) (by `diegosouzapw`) is an open-source, local-first AI gateway that aggregates hundreds of LLM providers into a single OpenAI-compatible API (`http://localhost:20128/v1`). It provides automatic failover, combo chaining, latency tracking, and local observability with zero telemetry.

The complete upstream distribution is housed intact inside `omiroute/upstream/OmniRoute` without any stripped or deleted features.

---

## 3. What Harshu AI OS Owns vs What `omiroute` Owns

```
+-------------------------------------------------------------+
|                       HARSHU AI OS                          |
|  - FastAPI Endpoints (/ask, /ask/rag)                       |
|  - High-level Question Classification Logic                |
|  - ChromaDB Vector Store & Distance Thresholding           |
|  - Chunk Ingestion, Token Slicing & BM25 Scoring           |
|  - Sufficiency Workflow & Grounding Decision Logic          |
|  - DDGS Web Search Execution & URL Parsing                  |
|  - Frontend Web UI & Latency Breakdown Display             |
|  - Business & Application Rules                             |
+-------------------------------------------------------------+
                              |
                     (Model Role Request)
                              v
+-------------------------------------------------------------+
|                         OMIROUTE                            |
|  - Local Gateway Daemon (http://localhost:20128/v1)         |
|  - Logical Role -> Provider/Model Translation               |
|  - Automatic Fallback on 429 Quota / 503 Provider Down      |
|  - OpenAI-Compatible Function/Tool Schema Preservation      |
|  - Embedding Model Isolation (Strict Dimension Enforcement) |
|  - Gateway Health & Provider Cooldown Monitoring            |
+-------------------------------------------------------------+
```

---

## 4. Architecture

```text
Harshu AI OS
      |
      | classification / judge / generation /
      | embedding / tool-capable model calls
      v
omiroute/
      | (http://localhost:20128/v1)
      v
OmniRoute Gateway
      |
      +--> Groq Cloud (Primary) ----[429 Quota Exceeded]----+
      |                                                     |
      +--> Google Gemini (Fallback 1) <---------------------+ (Failover)
      |
      +--> Cerebras Inference (Fallback 2)
```

---

## 5. Folder Structure

```text
omiroute/
├── upstream/             # Complete upstream OmniRoute distribution & package definition
│   ├── OmniRoute/        # Complete cloned upstream git repository (11,600+ files)
│   └── package.json      # Pinned npm runtime configuration
├── config/               # Harshu routing, combos, and provider metadata
│   ├── roles.json        # Logical roles, fallback chains, and capabilities
│   ├── combos.json       # OmniRoute combo configuration structure
│   └── providers.json    # Legitimate providers catalog
├── client/               # Python client adapter
│   ├── __init__.py
│   └── gateway_client.py # Standardized client for Harshu AI OS
├── scripts/              # PowerShell operational automation
│   ├── start.ps1         # Starts gateway daemon
│   ├── stop.ps1          # Stops gateway daemon
│   ├── health.ps1        # Gateway health check
│   └── smoke_test.ps1    # Runs smoke test suite
├── tests/                # Test suite
│   ├── __init__.py
│   ├── run_all_smoke_tests.py
│   ├── test_roles.py
│   └── test_memory.py
├── .env.example          # Placeholders for legitimate API keys
├── .gitignore            # Isolated ignore rules for node_modules and sqlite db
├── VERSION.md            # Upstream version and commit SHA reference
└── README.md             # This documentation
```

---

## 6. Upstream Version & Installation

- **Upstream Project:** [OmniRoute by diegosouzapw](https://github.com/diegosouzapw/OmniRoute)
- **Release Version:** `3.8.49`
- **Tracked Upstream Commit SHA:** `7837e469080b3355bbf30ee3f8e6b07c7f179a8d` (and release tag `v3.8.49` commit `c9d4a45f1883d7daf150bbff631f3e83b41aa5b4`)

### Installation

#### Option A: Subsystem Local (Recommended)
```bash
cd omiroute/upstream
npm install
```

#### Option B: Global CLI
```bash
npm install -g omniroute@3.8.49
```

---

## 7. Operations & Scripts

| Operation | Command | Description |
|---|---|---|
| **Start Gateway** | `powershell -File omiroute/scripts/start.ps1` | Launches OmniRoute on `http://127.0.0.1:20128` |
| **Stop Gateway** | `powershell -File omiroute/scripts/stop.ps1` | Gracefully terminates gateway process |
| **Health Check** | `powershell -File omiroute/scripts/health.ps1` | Queries gateway endpoint and models list |
| **Smoke Tests** | `powershell -File omiroute/scripts/smoke_test.ps1` | Runs all 8 mandatory subsystem smoke tests |

---

## 8. Environment Variables

Create `omiroute/.env` by copying `omiroute/.env.example`:

```bash
Copy-Item omiroute/.env.example omiroute/.env
```

| Variable | Description | Example |
|---|---|---|
| `OMNIROUTE_PORT` | Local gateway port | `20128` |
| `OMNIROUTE_HOST` | Local gateway binding address | `127.0.0.1` |
| `GROQ_API_KEY` | Groq Cloud API Key | `gsk_...` |
| `GEMINI_API_KEY` | Google Gemini API Key | `AIza...` |
| `CEREBRAS_API_KEY` | Cerebras Inference API Key | `csk_...` |
| `COHERE_API_KEY` | Cohere API Key (for embeddings) | `...` |

---

## 9. Providers Configured

Only legitimate, ToS-compliant API providers are configured:

1. **Groq Cloud:** Ultra-low latency inference for Llama 3.1 8B, Llama 3.3 70B, and GPT-OSS reasoning models.
2. **Google Gemini:** Reliable structured JSON generation, general text completion, and multimodal comprehension.
3. **Cerebras Inference:** High-throughput wafer-scale engine for high-speed fallback.
4. **Cohere AI:** High-accuracy semantic embedding vectors (1024 dimensions).

---

## 10. Logical Model Roles & Fallback Chains

| Logical Role | Purpose | Primary Model | Fallback Tier 1 | Fallback Tier 2 |
|---|---|---|---|---|
| `harshu-classifier` | Request classification | `groq/llama-3.1-8b-instant` | `gemini/gemini-2.5-flash` | `cerebras/llama-3.3-70b` |
| `harshu-judge` | RAG sufficiency judge | `gemini/gemini-2.5-flash` | `groq/llama-3.3-70b-versatile` | `cerebras/llama-3.3-70b` |
| `harshu-general` | Conversational generation | `gemini/gemini-2.5-flash` | `groq/llama-3.3-70b-versatile` | `groq/openai/gpt-oss-20b` |
| `harshu-reasoning` | Deep logic / architecture | `groq/openai/gpt-oss-120b` | `groq/deepseek-r1-distill-llama-70b` | `gemini/gemini-2.5-flash` |
| `harshu-tools` | Function calling (`web_search`) | `groq/llama-3.3-70b-versatile` | `gemini/gemini-2.5-flash` | `cerebras/llama-3.3-70b` |
| `harshu-embedding` | Vector generation | `cohere/embed-english-v3.0` | **NONE** *(Strict isolation)* | **NONE** |

---

## 11. Embedding Compatibility Rules

> [!CAUTION]
> **Strict Dimension & Model Isolation:**
> Vector embeddings from different model architectures cannot be co-indexed into the same ChromaDB collection without destroying semantic distance geometry.
> 
> Therefore, `harshu-embedding` uses a **Strict Primary-Only** strategy. If the primary embedding provider fails, the gateway will fail fast with a clear error rather than silently returning an incompatible vector.

---

## 12. Tool Calling Preservation

When Harshu AI OS calls `harshu-tools`:
- Harshu AI OS passes standard function definitions (e.g. `web_search`).
- The gateway passes the schema untouched to the model.
- Every model in the `harshu-tools` fallback chain is verified to support native function calling.
- If the model emits a tool call, `omiroute` returns the `id`, `name`, and `arguments` directly to Harshu AI OS.
- **The gateway NEVER executes external tools**; execution remains strictly inside Harshu AI OS.

---

## 13. Security & Terms-of-Service Decisions

To protect credentials and ensure long-term stability:
- **NO Account Scraping:** We do not configure ChatGPT Plus / Claude subscription cookie extraction or MITM interception.
- **Localhost Only:** The gateway binds strictly to `127.0.0.1`.
- **Zero Real Secrets in Code:** Real API keys stay in untracked `.env` files.

---

## 14. Memory Capability for Future Harshu AI OS Integration

Inspection of the pinned upstream OmniRoute distribution (`src/lib/memory/` and `src/app/api/memory/`) confirms a fully implemented, production-grade memory subsystem. Below is the complete factual breakdown of what exists in upstream OmniRoute:

### A. Memory Storage & Persistence
- **Primary Backend (`sqliteBackend.ts`):** Embedded SQLite database with full ACID persistence.
  - **Full-Text Search:** Native SQLite `FTS5` virtual table indexing for keyword queries.
  - **Vector Search:** `sqlite-vec` extension for local vector similarity search.
- **Secondary Backends:** Configurable pluggable backends for external Qdrant (`qdrant.ts`) and Obsidian vaults (`obsidianBackend.ts`).
- **Memory Types (`types.ts`):**
  - `factual`: User preferences, explicit facts, environment constraints.
  - `episodic`: Chronological conversation events, user milestones.
  - `procedural`: Coding conventions, project instructions, operational guidelines.
  - `semantic`: General concepts, domain terminology.

### B. Extraction & Summarization
- **Fact Extraction (`extraction.ts`):** Heuristic and model-assisted fact extraction pipeline that detects explicit preferences and profile traits.
- **Controlled Extraction Policy:** Automatic extraction is **strictly disabled by default**; extraction runs only when explicitly triggered.
- **Temporal Decay (`typedDecay.ts`):** Exponential decay scoring reduces the relevance of stale episodic memories over time.
- **Summarization (`summarization.ts`):** Compaction routine merging memories older than `olderThanDays` into dense summaries.

### C. Retrieval & Context Injection
- **Multi-Strategy Retrieval (`retrieval.ts`):**
  - `exact`: Pure FTS5 keyword matching.
  - `semantic`: Vector distance calculation (`sqlite-vec` / Qdrant).
  - `hybrid`: Reciprocal Rank Fusion (RRF) combining keyword and dense vector scores.
  - Optional cross-encoder reranking (`rerankEnabled`).
- **Context Injection (`injection.ts`):** Injects matched memories into downstream model calls within a strictly capped context token budget (`maxTokens`, default 2000 tokens) using structured `<memory>` tags.

### D. REST API Endpoints & Dashboard
- `POST /api/memory`: Store a memory record (`{ type, key, content, metadata }`).
- `GET /api/memory`: List stored memories with pagination and keyword filtering.
- `PUT /api/memory/[id]`: Edit content, key, or metadata.
- `DELETE /api/memory/[id]`: Explicit deletion for user privacy compliance.
- `POST /api/memory/retrieve-preview`: Interactive retrieval preview matching a query.
- `POST /api/memory/summarize`: Trigger memory compaction.
- `POST /api/memory/reindex`: Recompute embeddings across all stored items.
- `GET /api/memory/engine-status`: Real-time health metrics of FTS5, vector store, and cache.
- **Dashboard UI:** Dedicated Memory pane in the OmniRoute web interface (`http://localhost:20128`) for visual browsing, editing, and deleting records.

### E. Future Architecture with Harshu AI OS

> [!IMPORTANT]
> **Controlled Memory Boundary:**
> Harshu AI OS will **never automatically scrape or extract every conversational turn**.
> 
> Instead, Harshu AI OS maintains architectural ownership:
> 1. **Harshu AI OS decides WHAT should be remembered and WHEN retrieval occurs** (e.g. via explicit `/remember` commands or user profile preferences).
> 2. **OmniRoute provides the storage, vector indexing, and hybrid retrieval infrastructure.**

---

## 15. Available OmniRoute Capabilities Not Yet Integrated

The upstream OmniRoute distribution inside `omiroute/upstream/OmniRoute` is complete and unstripped. While Harshu AI OS currently configures only the model/provider gateway, the following major upstream capabilities remain available for future learning checkpoints:

1. **Model Context Protocol (MCP) Server Host (`@modelcontextprotocol/sdk`):**
   - *What it is:* Built-in support to connect external MCP tool servers (filesystem, databases, web search, GitHub) directly to models.
   - *Future value for Harshu AI OS:* Enables Harshu AI OS to dynamically expose local tools and resources through open standards without writing custom Python adapter schemas.

2. **Agent-to-Agent (A2A) Protocols & Multi-Agent Routing:**
   - *What it is:* Upstream routing layers for multi-agent panels, consensus voting, and agent debates.
   - *Future value for Harshu AI OS:* Can power multi-model verification where multiple LLMs critique a RAG answer or collaborate on code generation.

3. **Prompt & Semantic Response Caching (`ioredis` / in-memory cache):**
   - *What it is:* Intelligent caching that matches repeated user queries or identical prompt prefixes to return instantaneous responses.
   - *Future value for Harshu AI OS:* Speeds up repeated document questions and reduces token costs for static RAG system prompts.

4. **Token Compression Engines (RTK, Caveman, OmniGlyph):**
   - *What it is:* Heuristic and semantic context compressors designed to shrink prompt size by 15% to 95%.
   - *Future value for Harshu AI OS:* Allows fitting dozens of retrieved RAG document chunks into small-context or high-speed model windows without truncation.

5. **Multimodal OCR & Vision Processing (`/v1/ocr`):**
   - *What it is:* Dedicated transformation pipeline supporting Vertex DeepSeek-OCR, Mistral OCR, and Azure Document Intelligence.
   - *Future value for Harshu AI OS:* Enables RAG ingestion of scanned PDFs, diagrams, architecture flowcharts, and screenshots.

6. **Automated Evaluation Suites (`promptfoo` & `/api/evals`):**
   - *What it is:* Automated benchmark runner and prompt testing framework.
   - *Future value for Harshu AI OS:* Can integrate with Harshu's RAG Arena to evaluate model output quality and judge accuracy systematically.

---

## 16. Smoke Tests

Run all tests with:
```bash
uv run python omiroute/tests/run_all_smoke_tests.py
```

Tests verify:
1. `harshu-classifier` structured JSON output
2. `harshu-judge` sufficiency evaluation contract
3. `harshu-general` conversational response
4. `harshu-reasoning` logic analysis
5. `harshu-tools` function calling preservation (`get_game_server_status`)
6. `harshu-embedding` vector dimensions (1024-dim)
7. Deterministic failover mechanics (`429 -> Fallback -> 200 OK`)
8. Native tool-calling capability verification across all models in `harshu-tools`
9. Isolated memory capability contract (store synthetic preference + retrieve match)

---

## 17. Future Integration Plan for Harshu AI OS

When ready to connect Harshu AI OS to `omiroute`:
1. Point `LiteLLM` or standard `httpx` base URL in Harshu AI OS to `http://127.0.0.1:20128/v1`.
2. Replace hardcoded model strings (`"gemini/gemini-2.5-flash"`, `"groq/openai/gpt-oss-20b"`) with role aliases (`"harshu-classifier"`, `"harshu-general"`, etc.).
3. Enjoy automatic, transparent failover across all application routes.
