"""Orchestrator for the RAG Evaluation Arena."""

import argparse
import time
import os
import psutil
import re
from typing import List, Dict, Any

from harshu_ai_os.evaluations.arena_data import generate_documents, generate_queries, chunk_document_and_map_queries
from harshu_ai_os.evaluations.local_embeddings import embed_texts, embed_query

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024 / 1024:.2f} MB (RSS)"

def bm25_tokenize(text: str) -> List[str]:
    # Matches alphanumeric words, allowing internal hyphens and periods (e.g. v2.0, v3.1-beta, product-name)
    return re.findall(r'\b\w+(?:[-.]\w+)*\b', text.lower())

def rrf_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    # Sort by score (descending), then by doc_id (ascending) for determinism
    return [doc_id for doc_id, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]

def arena_upsert(collection, records: List[Dict[str, Any]], batch_size: int = 500):
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_ids = [r["id"] for r in batch]
        batch_docs = [r["text"] for r in batch]
        
        batch_metas = [{
            "source": r["source"], 
            "chunk_index": r["chunk_index"],
            "doc_id": r["doc_id"],
            "topic": r["topic"],
            "version": r["version"],
        } for r in batch]
        
        batch_embeddings = embed_texts(batch_docs)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
        )
        print(f"      Embedded and inserted {min(i + batch_size, len(records))}/{len(records)} chunks...")


def evaluate_arena(collection, queries, top_k: int = 5, use_reranker: bool = False, bm25_index=None, bm25_doc_ids=None) -> List[Dict[str, Any]]:
    if use_reranker:
        from harshu_ai_os.rag.reranker import rerank_candidates, get_reranker_model
        print("   Warming up CrossEncoder reranker...")
        warmup_start = time.perf_counter()
        # Warmup
        get_reranker_model()
        rerank_candidates("warmup query", {
            "ids": ["id1"], 
            "texts": ["text1"], 
            "distances": [0.1], 
            "metadatas": [{}]
        }, top_k=1)
        warmup_time = time.perf_counter() - warmup_start
        print(f"   Reranker warm-up took {warmup_time:.2f}s")
        
    results = []
    
    for idx, q in enumerate(queries):
        q_emb = embed_query(q.question)
        
        # We always retrieve top 50 once
        start_dense = time.perf_counter()
        
        retrieved_data = collection.query(
            query_embeddings=[q_emb],
            n_results=50,
            include=["documents", "metadatas", "distances"]
        )
        
        dense_latency = time.perf_counter() - start_dense
        
        retrieved_texts = retrieved_data["documents"][0] if retrieved_data["documents"] else []
        retrieved_metas = retrieved_data["metadatas"][0] if retrieved_data["metadatas"] else []
        retrieved_ids = retrieved_data["ids"][0] if retrieved_data["ids"] else []
        retrieved_dists = retrieved_data["distances"][0] if retrieved_data["distances"] else []
        
        # Dense Baseline
        dense_top_k = retrieved_ids[:top_k]
        
        rerank_latency = 0.0
        reranked_top_k = []
        
        bm25_latency = 0.0
        bm25_top_50 = []
        hybrid_latency = 0.0
        hybrid_top_50 = []
        
        if bm25_index is not None and bm25_doc_ids is not None:
            start_bm25 = time.perf_counter()
            tokenized_query = bm25_tokenize(q.question)
            doc_scores = bm25_index.get_scores(tokenized_query)
            
            top_50_indices = sorted(range(len(doc_scores)), key=lambda i: (-doc_scores[i], bm25_doc_ids[i]))[:50]
            bm25_top_50 = [bm25_doc_ids[i] for i in top_50_indices]
            bm25_latency = time.perf_counter() - start_bm25
            
            start_hybrid = time.perf_counter()
            hybrid_top_50 = rrf_fusion([retrieved_ids, bm25_top_50], k=60)
            hybrid_latency = time.perf_counter() - start_hybrid
            
        bm25_top_k = bm25_top_50[:top_k] if bm25_top_50 else []
        hybrid_top_k = hybrid_top_50[:top_k] if hybrid_top_50 else []
        
        if use_reranker and retrieved_texts:
            candidates = {
                "ids": retrieved_ids,
                "texts": retrieved_texts,
                "distances": retrieved_dists,
                "metadatas": retrieved_metas
            }
            start_rerank = time.perf_counter()
            ranked = rerank_candidates(q.question, candidates, top_k=top_k)
            rerank_latency = time.perf_counter() - start_rerank
            reranked_top_k = ranked["ids"]
        
        # Determine error states
        def get_error_type(eval_ids, top_50_ids):
            if not q.answerable:
                return None
            
            all_evidence_met = True
            for fact_id, chunks in q.expected_facts_to_chunks.items():
                if not any(cid in eval_ids for cid in chunks):
                    all_evidence_met = False
                    break
                    
            if not all_evidence_met:
                all_evidence_in_50 = True
                for fact_id, chunks in q.expected_facts_to_chunks.items():
                    if not any(cid in top_50_ids for cid in chunks):
                        all_evidence_in_50 = False
                        break
                return "ranking_failure" if all_evidence_in_50 else "retrieval_miss"
            return None # success
            
        dense_error = get_error_type(dense_top_k, retrieved_ids)
        rerank_error = get_error_type(reranked_top_k, retrieved_ids) if use_reranker else None
        bm25_error = get_error_type(bm25_top_k, bm25_top_50) if bm25_index else None
        hybrid_error = get_error_type(hybrid_top_k, hybrid_top_50) if bm25_index else None
        
        results.append({
            "id": q.id,
            "question": q.question,
            "category": q.category,
            "answerable": q.answerable,
            "dense_latency": dense_latency,
            "rerank_latency": rerank_latency,
            "bm25_latency": bm25_latency,
            "hybrid_latency": dense_latency + bm25_latency + hybrid_latency,
            "dense_ids": dense_top_k,
            "rerank_ids": reranked_top_k,
            "bm25_ids": bm25_top_k,
            "hybrid_ids": hybrid_top_k,
            "dense_error": dense_error,
            "rerank_error": rerank_error,
            "bm25_error": bm25_error,
            "hybrid_error": hybrid_error,
            "retrieved_50_ids": retrieved_ids,
            "expected_facts_to_chunks": q.expected_facts_to_chunks
        })
        
        if (idx + 1) % 50 == 0:
            print(f"   Evaluated {idx + 1}/{len(queries)} queries...")
            
    return results

def compute_metrics(results: List[Dict[str, Any]], key_ids: str):
    from harshu_ai_os.evaluations.retrieval_metrics import (
        calculate_hit_rate, calculate_mrr, calculate_precision_at_k, calculate_fact_recall_at_k
    )
    
    answerable = [r for r in results if r["answerable"]]
    
    ranks = []
    all_evidence_5_count = 0
    p5_list = []
    r5_list = []
    
    for r in answerable:
        all_expected_chunks = set()
        for chunks in r["expected_facts_to_chunks"].values():
            all_expected_chunks.update(chunks)
            
        rank = None
        for i, cid in enumerate(r[key_ids]):
            if cid in all_expected_chunks:
                rank = i + 1
                break
        ranks.append(rank)
        
        all_met = True
        for fact_id, chunks in r["expected_facts_to_chunks"].items():
            if not any(cid in r[key_ids] for cid in chunks):
                all_met = False
                break
        if all_met:
            all_evidence_5_count += 1
            
        if all_expected_chunks:
            p = calculate_precision_at_k(r[key_ids], list(all_expected_chunks), k=5)
            rec = calculate_fact_recall_at_k(r[key_ids], r["expected_facts_to_chunks"], k=5)
            p5_list.append(p)
            r5_list.append(rec)
            
    hit_1 = calculate_hit_rate(ranks, k=1)
    hit_3 = calculate_hit_rate(ranks, k=3)
    hit_5 = calculate_hit_rate(ranks, k=5)
    mrr = calculate_mrr(ranks)
    p5 = sum(p5_list) / len(p5_list) if p5_list else 0.0
    r5 = sum(r5_list) / len(r5_list) if r5_list else 0.0
    all_ev_5 = (all_evidence_5_count / len(answerable)) * 100 if answerable else 0.0
    
    return {
        "hit1": hit_1,
        "hit3": hit_3,
        "hit5": hit_5,
        "all_ev_5": all_ev_5,
        "mrr": mrr,
        "p5": p5,
        "fact_r5": r5
    }

def print_arena_metrics(results: List[Dict[str, Any]], use_reranker: bool, use_hybrid: bool):
    answerable = [r for r in results if r["answerable"]]
    unsupported = [r for r in results if not r["answerable"]]
    total = len(results)
    
    categories = {}
    for r in results:
        cat = r["category"]
        categories[cat] = categories.get(cat, 0) + 1
        
    print(f"\n=== ARENA METRICS (Side-by-Side Rematch) ===")
    print(f"Total Queries Evaluated : {total} ({len(answerable)} Supported, {len(unsupported)} Unsupported)")
    
    dense_metrics = compute_metrics(results, "dense_ids")
    if use_reranker:
        rerank_metrics = compute_metrics(results, "rerank_ids")
    if use_hybrid:
        bm25_metrics = compute_metrics(results, "bm25_ids")
        hybrid_metrics = compute_metrics(results, "hybrid_ids")
    
    print("\n--- RETRIEVAL METRICS ---")
    if use_hybrid:
        print(f"{'Metric':<25} | {'Dense Top-5':<15} | {'BM25 Top-5':<15} | {'Hybrid Top-5':<15}")
        print("-" * 75)
        print(f"{'Hit@1':<25} | {dense_metrics['hit1']:<14.2f}% | {bm25_metrics['hit1']:<14.2f}% | {hybrid_metrics['hit1']:<14.2f}%")
        print(f"{'Hit@3':<25} | {dense_metrics['hit3']:<14.2f}% | {bm25_metrics['hit3']:<14.2f}% | {hybrid_metrics['hit3']:<14.2f}%")
        print(f"{'Hit@5':<25} | {dense_metrics['hit5']:<14.2f}% | {bm25_metrics['hit5']:<14.2f}% | {hybrid_metrics['hit5']:<14.2f}%")
        print(f"{'AllEvidence@5':<25} | {dense_metrics['all_ev_5']:<14.2f}% | {bm25_metrics['all_ev_5']:<14.2f}% | {hybrid_metrics['all_ev_5']:<14.2f}%")
        print(f"{'MRR':<25} | {dense_metrics['mrr']:<15.4f} | {bm25_metrics['mrr']:<15.4f} | {hybrid_metrics['mrr']:<15.4f}")
        print(f"{'Context Precision@5':<25} | {dense_metrics['p5']:<15.4f} | {bm25_metrics['p5']:<15.4f} | {hybrid_metrics['p5']:<15.4f}")
        print(f"{'Fact Recall@5':<25} | {dense_metrics['fact_r5']:<15.4f} | {bm25_metrics['fact_r5']:<15.4f} | {hybrid_metrics['fact_r5']:<15.4f}")
    elif use_reranker:
        print(f"{'Metric':<25} | {'Dense Top-5':<15} | {'Reranked Top-5':<15}")
        print("-" * 60)
        print(f"{'Hit@1':<25} | {dense_metrics['hit1']:<14.2f}% | {rerank_metrics['hit1']:<14.2f}%")
        print(f"{'Hit@3':<25} | {dense_metrics['hit3']:<14.2f}% | {rerank_metrics['hit3']:<14.2f}%")
        print(f"{'Hit@5':<25} | {dense_metrics['hit5']:<14.2f}% | {rerank_metrics['hit5']:<14.2f}%")
        print(f"{'AllEvidence@5':<25} | {dense_metrics['all_ev_5']:<14.2f}% | {rerank_metrics['all_ev_5']:<14.2f}%")
        print(f"{'MRR':<25} | {dense_metrics['mrr']:<15.4f} | {rerank_metrics['mrr']:<15.4f}")
        print(f"{'Context Precision@5':<25} | {dense_metrics['p5']:<15.4f} | {rerank_metrics['p5']:<15.4f}")
        print(f"{'Fact Recall@5':<25} | {dense_metrics['fact_r5']:<15.4f} | {rerank_metrics['fact_r5']:<15.4f}")
    else:
        print(f"Hit@1                   : {dense_metrics['hit1']:.2f}%")
        print(f"Hit@5                   : {dense_metrics['hit5']:.2f}%")
        print(f"AllEvidence@5           : {dense_metrics['all_ev_5']:.2f}%")
        print(f"MRR                     : {dense_metrics['mrr']:.4f}")
        print(f"Context Precision@5     : {dense_metrics['p5']:.4f}")
        print(f"Fact Recall@5           : {dense_metrics['fact_r5']:.4f}")

    if use_hybrid or use_reranker:
        target_error = "hybrid_error" if use_hybrid else "rerank_error"
        rescued_rank = 0
        rescued_miss = 0
        regressions = 0
        dense_misses = 0
        dense_rank_fails = 0
        
        rescued_by_cat = {cat: 0 for cat in categories}
        regression_by_cat = {cat: 0 for cat in categories}
        
        rescued_examples = []
        regressed_examples = []
        
        dense_top50_misses = 0
        dense_top5_failures = 0
        bm25_top50_misses = 0
        bm25_top5_failures = 0
        hybrid_top50_misses = 0
        hybrid_top5_failures = 0
        hybrid_union_rescued_top50 = 0
        
        for r in answerable:
            dense_error = r["dense_error"]
            if dense_error == "retrieval_miss":
                dense_top50_misses += 1
            elif dense_error == "ranking_failure":
                dense_top5_failures += 1
                
            if use_hybrid:
                bm25_error = r.get("bm25_error")
                hybrid_error = r.get("hybrid_error")
                if bm25_error == "retrieval_miss":
                    bm25_top50_misses += 1
                elif bm25_error == "ranking_failure":
                    bm25_top5_failures += 1
                if hybrid_error == "retrieval_miss":
                    hybrid_top50_misses += 1
                elif hybrid_error == "ranking_failure":
                    hybrid_top5_failures += 1
                if dense_error == "retrieval_miss" and hybrid_error != "retrieval_miss":
                    hybrid_union_rescued_top50 += 1
            
            if r["dense_error"] == "retrieval_miss":
                dense_misses += 1
                if r[target_error] is None:
                    rescued_miss += 1
                    rescued_by_cat[r["category"]] += 1
                    rescued_examples.append(r)
            elif r["dense_error"] == "ranking_failure":
                dense_rank_fails += 1
                if r[target_error] is None:
                    rescued_rank += 1
                    rescued_by_cat[r["category"]] += 1
                    rescued_examples.append(r)
            
            if r["dense_error"] is None and r[target_error] is not None:
                regressions += 1
                regression_by_cat[r["category"]] += 1
                regressed_examples.append(r)
                
        print("\n--- EXPERIMENT IMPACT (Supported Queries) ---")
        if use_hybrid:
            print(f"Dense Misses/Failures : {dense_top50_misses} / {dense_top5_failures}")
            print(f"BM25  Misses/Failures : {bm25_top50_misses} / {bm25_top5_failures}")
            print(f"Hybrid Misses/Failures: {hybrid_top50_misses} / {hybrid_top5_failures}")
            print(f"Hybrid Union rescued Top-50 missing evidence: {hybrid_union_rescued_top50}")
        else:
            print(f"Dense Retrieval Misses (Not in top 50) : {dense_misses}")
            print(f"Dense Ranking Failures (In top 50)     : {dense_rank_fails}")
        print("-" * 50)
        print(f"Rescued Ranking Failures (Success)     : +{rescued_rank}")
        print(f"Rescued Retrieval Misses (Success)     : +{rescued_miss}")
        print(f"Regressions (New Failures)             : -{regressions}")
        print(f"Net Change in Successful Queries       : {rescued_rank + rescued_miss - regressions:+} queries")
        
        print("\n--- IMPACT BY CATEGORY ---")
        print(f"{'Category':<18} | {'Rescued':<10} | {'Regressed':<10}")
        print("-" * 45)
        for cat in categories:
            res = rescued_by_cat[cat]
            reg = regression_by_cat[cat]
            if res > 0 or reg > 0:
                print(f"{cat:<18} | +{res:<9} | -{reg:<9}")
                
        print("\n--- NOTABLE EXAMPLES ---")
        if rescued_examples:
            print("Examples of Rescued Queries:")
            for r in rescued_examples[:3]:
                print(f"  - [{r['category']}] {r['question']}")
        if regressed_examples:
            print("Examples of Regressions:")
            for r in regressed_examples[:3]:
                print(f"  - [{r['category']}] {r['question']}")
    
    dense_lats = sorted([r["dense_latency"] for r in results])
    p50_d = dense_lats[int((total - 1) * 0.50)]
    p95_d = dense_lats[int((total - 1) * 0.95)]
    p99_d = dense_lats[int((total - 1) * 0.99)]
    
    if use_hybrid:
        bm25_lats = sorted([r["bm25_latency"] for r in results])
        p50_b = bm25_lats[int((total - 1) * 0.50)]
        p95_b = bm25_lats[int((total - 1) * 0.95)]
        p99_b = bm25_lats[int((total - 1) * 0.99)]
        
        tot_lats = sorted([r["hybrid_latency"] for r in results])
        p50_t = tot_lats[int((total - 1) * 0.50)]
        p95_t = tot_lats[int((total - 1) * 0.95)]
        p99_t = tot_lats[int((total - 1) * 0.99)]
        
        print("\n--- STEADY-STATE LATENCY ---")
        print(f"{'Percentile':<12} | {'Dense Top-50':<15} | {'BM25 Top-50':<15} | {'Hybrid Total Path':<15}")
        print("-" * 65)
        print(f"{'p50':<12} | {p50_d:<14.4f}s | {p50_b:<14.4f}s | {p50_t:<14.4f}s")
        print(f"{'p95':<12} | {p95_d:<14.4f}s | {p95_b:<14.4f}s | {p95_t:<14.4f}s")
        print(f"{'p99':<12} | {p99_d:<14.4f}s | {p99_b:<14.4f}s | {p99_t:<14.4f}s")
    elif use_reranker:
        rr_lats = sorted([r["rerank_latency"] for r in results])
        p50_r = rr_lats[int((total - 1) * 0.50)]
        p95_r = rr_lats[int((total - 1) * 0.95)]
        
        tot_lats = sorted([r["dense_latency"] + r["rerank_latency"] for r in results])
        p50_t = tot_lats[int((total - 1) * 0.50)]
        p95_t = tot_lats[int((total - 1) * 0.95)]
        
        print("\n--- STEADY-STATE LATENCY ---")
        print(f"{'Percentile':<12} | {'Dense Top-50':<15} | {'Reranking':<15} | {'Total Path':<15}")
        print("-" * 65)
        print(f"{'p50':<12} | {p50_d:<14.4f}s | {p50_r:<14.4f}s | {p50_t:<14.4f}s")
        print(f"{'p95':<12} | {p95_d:<14.4f}s | {p95_r:<14.4f}s | {p95_t:<14.4f}s")
    else:
        print("\n--- STEADY-STATE LATENCY (Dense Top-50) ---")
        print(f"p50: {p50_d:.4f}s | p95: {p95_d:.4f}s | p99: {p99_d:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Arena v1 (Local Embeddings)")
    parser.add_argument("--scale", type=int, default=5000, help="Target chunks to generate")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "holdout"], help="Which query split to evaluate")
    parser.add_argument("--rerank", action="store_true", help="Use CrossEncoder reranker")
    parser.add_argument("--hybrid", action="store_true", help="Use Dense + BM25 Hybrid evaluation")
    args = parser.parse_args()
    
    print(f"Initializing RAG Arena (Scale: ~{args.scale} chunks, Split: {args.split}, Rerank: {args.rerank}, Hybrid: {args.hybrid}, Model: nomic-ai/nomic-embed-text-v1.5)")
    
    print("1. Generating synthetic corpus...")
    docs, facts = generate_documents(target_chunks=args.scale)
    print("2. Generating and stratifying queries...")
    queries = generate_queries(facts)
    
    print("3. Chunking and mapping ground truth...")
    records = chunk_document_and_map_queries(docs, queries)
    
    dev_count = sum(1 for q in queries if q.split == "dev")
    holdout_count = sum(1 for q in queries if q.split == "holdout")
    
    print(f"   Generated {len(docs)} documents -> {len(records)} chunks.")
    print(f"   Total Queries: {len(queries)} (Dev: {dev_count}, Holdout: {holdout_count})")
    
    active_queries = [q for q in queries if q.split == args.split]
    print(f"   Evaluating {len(active_queries)} queries in '{args.split}' split.")
    
    bm25_index = None
    bm25_doc_ids = None
    if args.hybrid:
        print("4(a). Building BM25Okapi Index...")
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [bm25_tokenize(r["text"]) for r in records]
        bm25_index = BM25Okapi(tokenized_corpus)
        bm25_doc_ids = [r["id"] for r in records]
    
    print("4(b). Ingesting into temporary Chroma collection using local model...")
    import chromadb
    chroma_client = chromadb.Client()
    collection_name = f"arena_{args.scale}_{int(time.time())}"
    collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    
    arena_upsert(collection, records)
        
    print(f"   Ingestion complete. Process Memory: {get_memory_usage()}")
    
    print("5. Running retrieval evaluation...")
    results = evaluate_arena(collection, active_queries, top_k=5, use_reranker=args.rerank, bm25_index=bm25_index, bm25_doc_ids=bm25_doc_ids)
    
    print_arena_metrics(results, args.rerank, args.hybrid)
    
if __name__ == "__main__":
    main()
