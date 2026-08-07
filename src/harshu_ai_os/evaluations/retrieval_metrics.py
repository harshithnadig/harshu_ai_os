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
