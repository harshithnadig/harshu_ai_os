import pytest
from harshu_ai_os.evaluations.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_hit_rate,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_fact_recall_at_k
)

def test_calculate_hit_at_k():
    assert calculate_hit_at_k(1, 3) is True
    assert calculate_hit_at_k(3, 3) is True
    assert calculate_hit_at_k(4, 3) is False
    assert calculate_hit_at_k(None, 3) is False

def test_calculate_mrr():
    assert calculate_mrr([1, 2, None]) == (1 + 0.5 + 0) / 3
    assert calculate_mrr([]) == 0.0

def test_calculate_hit_rate():
    assert calculate_hit_rate([1, 4, None], 3) == (1 / 3) * 100
    assert calculate_hit_rate([], 3) == 0.0

def test_calculate_precision_at_k():
    assert calculate_precision_at_k(["c1", "c2", "c3"], ["c2", "c4"], 2) == 0.5
    assert calculate_precision_at_k(["c1", "c2"], ["c3"], 2) == 0.0
    assert calculate_precision_at_k([], ["c1"], 2) == 0.0

def test_calculate_recall_at_k():
    assert calculate_recall_at_k(["c1", "c2", "c3"], ["c2", "c4"], 3) == 0.5
    assert calculate_recall_at_k(["c1", "c2"], ["c3"], 2) == 0.0
    assert calculate_recall_at_k([], ["c1"], 2) == 0.0

def test_calculate_fact_recall_at_k():
    facts = {
        "f1": ["c1", "c2"],
        "f2": ["c3", "c4"],
        "f3": ["c5"]
    }
    # hit 2 out of 3 facts
    assert calculate_fact_recall_at_k(["c1", "c4", "c6"], facts, 3) == 2 / 3
    
    # hit all facts (multiple chunks for same fact doesn't double-count)
    assert calculate_fact_recall_at_k(["c1", "c2", "c5"], facts, 3) == 2 / 3 # wait, c1/c2 hit f1, c5 hits f3, total 2 facts. Correct.
    
    # empty retrieval
    assert calculate_fact_recall_at_k([], facts, 3) == 0.0

def test_rrf_fusion_disjoint():
    # Test RRF with completely disjoint rankings
    from harshu_ai_os.evaluations.run_arena import rrf_fusion
    r1 = ["a", "b", "c"]
    r2 = ["d", "e", "f"]
    fused = rrf_fusion([r1, r2], k=60)
    assert len(fused) == 6
    assert set(fused) == {"a", "b", "c", "d", "e", "f"}
    # Verify order: 'a' and 'd' are rank 0, score 1/61. Deterministic tiebreak: 'a' then 'd'.
    assert fused == ["a", "d", "b", "e", "c", "f"]

def test_rrf_fusion_overlapping():
    # Test RRF with duplicate/overlapping rankings
    from harshu_ai_os.evaluations.run_arena import rrf_fusion
    r1 = ["a", "b", "c"]
    r2 = ["c", "b", "d"]
    fused = rrf_fusion([r1, r2], k=60)
    # 'b' gets 1/62 + 1/62 = 2/62 = 0.0322
    # 'c' gets 1/63 + 1/61 = 0.0322
    # 'a' gets 1/61 = 0.0163
    # 'd' gets 1/63 = 0.0158
    assert len(fused) == 4
    # 'b' and 'c' are top. Let's verify deterministic sorting.
    assert "b" in fused[:2] and "c" in fused[:2]
    # 'a' is next, 'd' is last
    assert fused[2] == "a"
    assert fused[3] == "d"

def test_rrf_fusion_deterministic():
    # Test deterministic tie-breaking and fixed k=60
    from harshu_ai_os.evaluations.run_arena import rrf_fusion
    r1 = ["z"]
    r2 = ["a"]
    # Both at rank 0, score 1/61. Tie broken by string ascending: 'a' before 'z'
    fused = rrf_fusion([r1, r2], k=60)
    assert fused == ["a", "z"]
    fused_rev = rrf_fusion([r2, r1], k=60)
    assert fused_rev == ["a", "z"]
