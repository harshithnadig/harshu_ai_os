# Learning Now: RAG Evaluation Metrics

This page is your desk. Open one learning lane at a time; the rest of the repo
can stay closed even though the finished application still uses it.

## Open now: the three-file metrics desk

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

## Open next: optional reranking lab

Reranking is advanced RAG, but its job is simple:

```text
fast search -> possible chunks -> careful second ranking -> better top results
```

When Hit@k and MRR feel comfortable, open only these files:

1. `rag/reranker.py` — scores and reorders already retrieved candidates.
2. `tests/test_reranker.py` — proves fields stay attached while order changes.
3. `evaluations/evaluate_reranker.py` — compares quality and latency.

Small vocabulary:

- **candidate:** a chunk returned by the first search;
- **CrossEncoder:** a model that reads the question and one chunk together;
- **rerank:** change candidate order using the new relevance scores;
- **latency:** how long the extra step takes.

Install this optional lab only when you are ready:

```powershell
uv sync --extra reranking
```

## Topic shelf

Distance gates, LLM sufficiency, provider retries, FastAPI schemas, ingestion,
and frontend rendering are preserved and tested. They are separate future
lanes—not prerequisites you must hold in your head today.
