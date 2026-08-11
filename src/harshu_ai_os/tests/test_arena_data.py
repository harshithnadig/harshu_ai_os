from harshu_ai_os.evaluations.arena_data import (
    generate_fact_pool,
    generate_documents,
    generate_queries,
    chunk_document_and_map_queries,
)

def test_fact_mapping_to_chunks():
    docs, facts = generate_documents(seed=42, target_chunks=20, chunk_size_words=10)
    
    queries = generate_queries(facts, seed=42, count=10)
    
    # We want to ensure that chunk mapper exactly finds facts
    records = chunk_document_and_map_queries(docs, queries, chunk_size=10)
    
    assert len(records) > 0
    
    # Verify that multi-evidence queries successfully mapped facts to chunks
    for q in queries:
        if q.answerable:
            assert len(q.expected_facts_to_chunks) == len(q.reference_fact_ids)
            for fact_id, chunks in q.expected_facts_to_chunks.items():
                assert isinstance(chunks, list)
                
def test_deterministic_generation():
    docs1, facts1 = generate_documents(seed=99, target_chunks=20)
    docs2, facts2 = generate_documents(seed=99, target_chunks=20)
    
    assert len(docs1) == len(docs2)
    assert docs1[0].text == docs2[0].text
    
    queries1 = generate_queries(facts1, seed=99, count=5)
    queries2 = generate_queries(facts2, seed=99, count=5)
    
    assert queries1[0].question == queries2[0].question
