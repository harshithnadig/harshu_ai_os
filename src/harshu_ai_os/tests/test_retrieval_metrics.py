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
    ranks = [1, 2, None, 5]

    result = calculate_hit_rate(ranks, 3)

    assert result == 50.0


def test_calculate_mrr():
    ranks = [1, 2, None, 5]

    result = calculate_mrr(ranks)

    expected = (1 / 1 + 1 / 2 + 0 + 1 / 5) / 4

    assert result == expected


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