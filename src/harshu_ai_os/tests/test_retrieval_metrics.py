from harshu_ai_os.evaluations.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_hit_rate,
    calculate_mrr,
    extract_ranks,
)


def test_calculate_hit_at_k():
    assert calculate_hit_at_k(2, 5) is True
    assert calculate_hit_at_k(6, 5) is False
    assert calculate_hit_at_k(None, 5) is False


def test_calculate_hit_rate():
    # This is the same small example used in the current learning checkpoint.
    ranks = [2, None, 4, 1]

    result = calculate_hit_rate(ranks, 3)

    # Only ranks 2 and 1 are inside the first three results: 2 / 4 = 50%.
    assert result == 50.0


def test_calculate_mrr():
    ranks = [2, None, 4, 1]

    result = calculate_mrr(ranks)

    # Each query contributes 1 / rank; a missing document contributes zero.
    expected = (1 / 2 + 0 + 1 / 4 + 1 / 1) / 4

    # The worked example must produce the exact MRR learned in the lesson.
    assert result == expected
    assert result == 0.4375


def test_extract_ranks():
    case_results = [
        {
            "evaluation": {
                "rank": 1
            }
        },
        {
            "evaluation": {
                "rank": 3
            }
        },
        {
            "evaluation": {
                "rank": None
            }
        },
    ]

    result = extract_ranks(case_results)

    assert result == [1, 3, None]
