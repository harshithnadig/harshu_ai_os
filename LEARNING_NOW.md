# Learning Now: RAG Evaluation Metrics

This is the only path you need for the current checkpoint.

## The three-file desk

Keep only these files open while learning metrics:

1. `src/harshu_ai_os/evaluations/retrieval_cases.py` — synthetic questions and expected evidence.
2. `src/harshu_ai_os/evaluations/retrieval_evaluator.py` — finds the relevant evidence and records its rank.
3. `src/harshu_ai_os/evaluations/retrieval_metrics.py` — turns ranks into Hit@k, hit rate, and MRR.

Everything under `api/`, `llm/`, `rag/`, and `frontend/` already supports the
application. You do not need to understand those folders during this lesson.

The earlier calculator and Big-O exercises now live only in the numbered Jupyter
learning notebooks. They are intentionally not part of the AI OS runtime.

## One mental model

```text
question -> retrieved chunks -> relevant rank -> metric

rank 2  -> hit at k=3 -> reciprocal rank 1/2
None    -> miss       -> reciprocal rank 0
rank 4  -> miss at 3  -> reciprocal rank 1/4
rank 1  -> hit at 3   -> reciprocal rank 1
```

For `[2, None, 4, 1]`:

- Hit@3 values are `[True, False, False, True]`.
- Hit rate at 3 is `2 / 4 = 50%`.
- MRR is `(1/2 + 0 + 1/4 + 1) / 4 = 0.4375`.

## Run only this lesson

```powershell
uv run pytest -q src/harshu_ai_os/tests/test_retrieval_metrics.py
```

Run the complete suite only when you are ready to check integration:

```powershell
uv run pytest -q
```

## Later, not now

- Distance thresholds and abstention
- LLM-based context sufficiency
- Provider routing and retries
- FastAPI response schemas
- Chroma ingestion
- Frontend rendering

These topics are preserved. They are simply closed until the current metric
checkpoint is understood.
