# RAG Arena v1 Results

## 1. Dataset
- **Documents:** 1,250
- **Chunks:** 5,866
- **Total Queries:** 500
- **DEV Split:** 397 queries (270 supported / 127 unsupported)
- **HOLDOUT Split:** 103 queries (70 supported / 33 unsupported)

## 2. Corrected DEV Dense Baseline
*Hit@1: 30.74% | Hit@3: 44.44% | Hit@5: 51.11% | AllEvidence@5: 40.37% | MRR: 0.3805 | Context Precision@5: 0.2704 | Fact Recall@5: 0.4574*
*Top-50 misses: 54 / Top-5 ranking failures: 107*

## 3. CrossEncoder Experiment
Tested a CrossEncoder re-ranker. It improved top-5 ranking of retrieved candidates but suffered from missing evidence at the initial Dense retrieval stage (top-50 starvation), leaving queries like `version_conflict` unresolvable. **Rejected** for production until top-50 candidate retrieval is fixed.

## 4. DEV Dense vs BM25 vs Hybrid Results
Explored BM25 and Hybrid search (Dense + BM25 with RRF). Hybrid successfully reduced Top-50 retrieval misses from 54 to 27. It rescued 23 ranking failures with only 5 regressions, achieving a net gain of 18 successful queries compared to Dense.

## 5. Frozen Architecture for HOLDOUT
The following architecture was frozen for final holdout validation:
- Dense candidate depth: 50
- BM25 candidate depth: 50
- RRF fusion (k=60)
- Top-5 final evaluation

## 6. HOLDOUT Results
**Dense / BM25 / Hybrid**

- **Hit@1**: 35.71% / 28.57% / 35.71%
- **Hit@3**: 51.43% / 60.00% / 54.29%
- **Hit@5**: 57.14% / 62.86% / 60.00%
- **AllEvidence@5**: 45.71% / 51.43% / 51.43%
- **MRR**: 0.4367 / 0.4271 / 0.4517
- **Context Precision@5**: 0.2943 / 0.2914 / 0.3229
- **Fact Recall@5**: 0.5143 / 0.5714 / 0.5571

### Hybrid vs Dense HOLDOUT Analysis:
- **Top-50 misses**: Dense 19 | BM25 13 | Hybrid 7
- **Top-5 ranking failures**: Dense 19 | BM25 21 | Hybrid 27
- **Ranking-failure rescues**: 6
- **Retrieval-miss-to-top5 rescues**: 0
- **Regressions**: 2
- **Net top-5 successes**: +4
- **Impact by Category (version_conflict)**: +5 rescued / -1 regressed

## 7. Conclusion
- BM25 produced the strongest Holdout Hit@3, Hit@5 and Fact Recall@5.
- Hybrid tied BM25 for AllEvidence@5 and produced the strongest MRR and Context Precision@5.
- Hybrid produced the strongest Top-50 candidate coverage, reducing Dense misses from 19 to 7.
- Therefore, Hybrid is the strongest candidate-generation strategy tested, but plain RRF is not conclusively the best final top-5 ranking strategy.

**Classification**: GENERALIZES WITH TRADE-OFFS.

*Limitations*: This experiment used synthetic data, local in-memory Nomic embeddings, and an evaluation environment that differs from production. No production improvements are claimed because `/ask/rag` remains unchanged.
