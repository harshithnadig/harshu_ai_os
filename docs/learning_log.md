## 2026-07-22 — Persistent Chroma-backed RAG integration

### What I built

I extended Harshu AI OS from manual in-memory similarity search toward a persistent RAG workflow using ChromaDB.

The system can now:

- store note IDs, documents, embeddings and metadata;
- embed an incoming question;
- retrieve the three closest stored notes;
- combine the retrieved notes into grounded context;
- send that context through the existing LLM routing and generation layer;
- expose a separate RAG API response containing the answer and retrieval evidence.

### What I learned

Manual RAG recalculates and compares note embeddings during every request.

ChromaDB stores document embeddings persistently, so later requests only need to embed the incoming question and search the existing vector collection.

Chroma returns distances rather than the similarity score used in the earlier manual implementation. With cosine distance, a smaller value represents a closer match.

The query result contains nested lists because Chroma supports querying multiple embeddings in one call, even when the current request contains only one question.

### Design decisions

The normal LLM endpoint and the RAG endpoint remain separate.

This keeps direct generation and grounded generation easier to debug, test and explain. The RAG endpoint also requires a richer response contract containing context, IDs, metadata and distances.

I built the retrieval mechanics directly before introducing LangChain so I can understand what the framework later abstracts.

### Evidence

Command:

`uv run pytest -v`

Result:

`18 passed, 3 warnings in 58.05s`

This proves that the existing tested behaviour still passes after the integration changes.

### Current limitation

The new RAG endpoint does not yet have direct automated coverage and has not yet been demonstrated end-to-end with seeded synthetic notes.

The next step is to run the endpoint, inspect the real response, add focused tests and handle empty or insufficient retrieval safely.