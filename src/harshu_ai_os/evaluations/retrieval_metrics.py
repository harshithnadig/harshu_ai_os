# This file contains the mathematical and logical metrics used to evaluate our system's performance.

def calculate_hit_at_k(rank, k):
    """Return True if rank <= k, else False"""
    if rank is None:
        return False

    return rank <= k


def calculate_mrr(ranks):
    """Mean Reciprocal Rank"""
    reciprocal_ranks = []

    if not ranks:
        return 0.0

    for rank in ranks:
        if rank is None:
            reciprocal_ranks.append(0)
        else:
            reciprocal_ranks.append(1 / rank)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def extract_ranks(case_results):
    """Extract all ranks from the results, skipping None values."""
    ranks = []
    for case in case_results:
        rank = case["evaluation"]["rank"]
        if rank is not None:
            ranks.append(rank)
        else:
            ranks.append(None)
    return ranks


def calculate_hit_rate(ranks, k):
    """Return the percentage of ranks that are <= k."""
    if not ranks:
        return 0.0

    hits = 0

    for r in ranks:
        if calculate_hit_at_k(r, k):
            hits += 1
    return (hits / len(ranks)) * 100
