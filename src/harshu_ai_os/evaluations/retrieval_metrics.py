"""Small, pure functions for the RAG metrics in the current lesson.

The functions know nothing about Chroma, APIs, or language models. They accept
ordinary Python values, which makes the maths easy to read and test.
"""


def calculate_hit_at_k(rank: int | None, k: int) -> bool:
    """Return whether the relevant document appeared in the first k results."""
    # None means retrieval never found the relevant document.
    if rank is None:
        return False

    # A smaller rank is better: rank 1 is the first retrieved result.
    return rank <= k


def calculate_mrr(ranks: list[int | None]) -> float:
    """Return Mean Reciprocal Rank, counting every missed query as zero."""
    # An empty evaluation has no score and must not divide by zero.
    if not ranks:
        return 0.0

    reciprocal_ranks = []

    for rank in ranks:
        # A miss earns no credit. A hit at rank n earns 1 / n credit.
        contribution = 0.0 if rank is None else 1 / rank
        reciprocal_ranks.append(contribution)

    # "Mean" means add every contribution and divide by query count.
    return sum(reciprocal_ranks) / len(ranks)


def extract_ranks(case_results: list[dict]) -> list[int | None]:
    """Read each recorded rank while preserving None for retrieval misses."""
    ranks = []

    for case in case_results:
        ranks.append(case["evaluation"]["rank"])

    return ranks


def calculate_hit_rate(ranks: list[int | None], k: int) -> float:
    """Return the percentage of evaluated queries that are hits at k."""
    if not ranks:
        return 0.0

    hits = 0

    for rank in ranks:
        if calculate_hit_at_k(rank, k):
            hits += 1

    # Multiply the fraction by 100 so 0.5 becomes the readable value 50.0.
    return (hits / len(ranks)) * 100


def calculate_precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Return the fraction of the top-k retrieved chunks that are relevant."""
    if not retrieved_ids or k <= 0:
        return 0.0

    considered_ids = retrieved_ids[:k]
    relevant_retrieved = 0

    for chunk_id in considered_ids:
        if chunk_id in expected_ids:
            relevant_retrieved += 1

    return relevant_retrieved / len(considered_ids)


def calculate_recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Return the fraction of expected relevant chunks that appear in the top-k results."""
    if not expected_ids or not retrieved_ids or k <= 0:
        return 0.0

    considered_ids = retrieved_ids[:k]
    relevant_retrieved = 0

    for expected_id in expected_ids:
        if expected_id in considered_ids:
            relevant_retrieved += 1

    return relevant_retrieved / len(expected_ids)

def calculate_fact_recall_at_k(retrieved_ids: list[str], expected_facts_to_chunks: dict[str, list[str]], k: int) -> float:
    """Return the fraction of expected unique facts that appear in the top-k results.
    
    This avoids penalizing the retriever for finding one valid chunk for a fact when 
    many equivalent chunks exist.
    """
    if not expected_facts_to_chunks or not retrieved_ids or k <= 0:
        return 0.0

    considered_ids = retrieved_ids[:k]
    facts_found = 0

    for fact_id, valid_chunks in expected_facts_to_chunks.items():
        # Check if ANY valid chunk for this fact is in the retrieved results
        if any(chunk_id in considered_ids for chunk_id in valid_chunks):
            facts_found += 1

    return facts_found / len(expected_facts_to_chunks)
