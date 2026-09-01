# Architecture Decision Records (ADRs) - Harshu AI OS V1

## ADR-001: Stateful Workflow Engine (LangGraph Evaluation)

### Status
**DECIDED: REJECTED FOR V1 (Preserve Pure Python Bounded ReAct)**

### Context
Evaluating whether to introduce LangGraph or comparable state-graph frameworks for Harshu AI OS V1 agent execution.

### Evaluation & Analysis
- **Current Architecture:** Harshu AI OS V1 uses a lightweight, bounded ReAct agent loop (`src/harshu_ai_os/agents/loop.py`) with explicit tool schemas, step bounds (`DEFAULT_MAX_STEPS = 5`), and deterministic step tracking (`AgentStep`).
- **Complexity Overhead:** LangGraph introduces state graphs, channel subscriptions, checkpointers, and heavy dependencies.
- **Requirement Fit:** Harshu AI OS V1 requests are synchronous HTTP queries without multi-tenant human-in-the-loop pause/resume or distributed graph state requirements.
- **Maintainability:** Pure Python ReAct execution is transparent, easily unit-tested with standard mocks, and imposes zero runtime overhead.

### Decision
Do not add LangGraph in V1. Retain the self-contained, bounded ReAct loop. Re-evaluate if V2 introduces long-running multi-turn sessions or asynchronous human review gates.

---

## ADR-002: External Persistence and Message Queues (Postgres, Redis, Celery)

### Status
**DECIDED: REJECTED FOR V1 (Maintain Zero-External-Dependency Runtime)**

### Context
Evaluating whether to integrate external relational databases (PostgreSQL), distributed caches (Redis), or asynchronous task brokers (Celery/RabbitMQ) into V1.

### Evaluation & Analysis
- **Vector Storage:** Local persistent ChromaDB (`chromadb.PersistentClient`) completely satisfies vector indexing, semantic search, and document embedding needs with zero external container requirements.
- **Rate Limiting:** The in-memory sliding-window limiter (`src/harshu_ai_os/api/security.py`) provides sub-millisecond, thread-safe rate enforcement without Redis overhead.
- **Request Tracing:** Pure ASGI middleware and Python `contextvars` manage request IDs and latency tracking without external logging databases.
- **Deployment Simplicity:** Requiring external database services increases cold-start times, causes local deployment failure modes, and complicates Docker Compose orchestrations.

### Decision
Do not introduce PostgreSQL, Redis, or Celery in V1. Keep the deployment lightweight, self-contained, and runnable in a single container or development process. Re-evaluate when distributed multi-node horizontal scaling or asynchronous background document pipelines are implemented in future versions.
