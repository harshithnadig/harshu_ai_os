from unittest.mock import MagicMock, patch

from harshu_ai_os.evaluations.retrieval_evaluator import (
    evaluate_abstention_thresholds,
    run_retrieval_evaluation,
)


@patch("harshu_ai_os.evaluations.retrieval_evaluator.query_notes")
def test_run_retrieval_evaluation_marks_unsupported_case_as_not_applicable(
    mock_query_notes,
):
    mock_collection = MagicMock()
    mock_client = MagicMock()

    evaluation_cases = [
        {
            "id": "rag_006",
            "question": "Which payment gateway does Harshu AI OS use?",
            "answerable": False,
            "expected_evidence": None,
        }
    ]

    results = run_retrieval_evaluation(mock_collection, mock_client, evaluation_cases)

    mock_query_notes.assert_not_called()
    assert len(results["results"]) == 1
    case_res = results["results"][0]
    assert case_res["evaluation"]["status"] == "not_applicable"
    assert case_res["evaluation"]["matched"] is None
    assert case_res["rank"] is None


@patch("harshu_ai_os.evaluations.retrieval_evaluator.query_notes")
def test_run_retrieval_evaluation_unsupported_cases_do_not_reduce_accuracy(
    mock_query_notes,
):
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_query_notes.return_value = {
        "texts": ["ChromaDB stores document embeddings."],
        "metadatas": [{"source": "overview.txt"}],
        "ids": ["overview-0"],
    }

    evaluation_cases = [
        {
            "id": "rag_001",
            "question": "What vector database does Harshu AI OS use?",
            "answerable": True,
            "expected_evidence": "Chroma",
        },
        {
            "id": "rag_006",
            "question": "Which payment gateway does Harshu AI OS use?",
            "answerable": False,
            "expected_evidence": None,
        },
    ]

    results = run_retrieval_evaluation(mock_collection, mock_client, evaluation_cases)

    summary = results["summary"]
    assert summary["total_cases"] == 2
    assert summary["answerable_cases"] == 1
    assert summary["unanswerable_cases"] == 1
    assert summary["retrieval_passed"] == 1
    assert summary["retrieval_failed"] == 0
    assert summary["retrieval_accuracy"] == 100.0


@patch("harshu_ai_os.evaluations.retrieval_evaluator.query_notes")
def test_evaluate_abstention_thresholds_computes_sweep_metrics(mock_query_notes):
    mock_collection = MagicMock()
    mock_client = MagicMock()

    # Case 1 (answerable): distances min 0.20
    # Case 2 (unanswerable): distances min 0.30
    def fake_query_notes(collection, client, question):
        if "vector" in question:
            return {"distances": [0.20, 0.40]}
        return {"distances": [0.30, 0.50]}

    mock_query_notes.side_effect = fake_query_notes

    cases = [
        {
            "id": "c1",
            "question": "vector",
            "answerable": True,
            "expected_evidence": "Chroma",
        },
        {
            "id": "c2",
            "question": "payment",
            "answerable": False,
            "expected_evidence": None,
        },
    ]

    sweep = evaluate_abstention_thresholds(
        mock_collection,
        mock_client,
        cases,
        candidate_thresholds=[0.15, 0.25, 0.35],
    )

    # Threshold 0.15: both min distances (0.20, 0.30) > 0.15 => abstains on both.
    # Correct Gen = 0, Correct Abs = 1, False Accept = 0, False Abs = 1
    assert sweep[0] == {
        "threshold": 0.15,
        "correct_generations": 0,
        "correct_abstentions": 1,
        "false_accepts": 0,
        "false_abstentions": 1,
    }

    # Threshold 0.25: c1 (0.20 <= 0.25) generates, c2 (0.30 > 0.25) abstains.
    # Correct Gen = 1, Correct Abs = 1, False Accept = 0, False Abs = 0
    assert sweep[1] == {
        "threshold": 0.25,
        "correct_generations": 1,
        "correct_abstentions": 1,
        "false_accepts": 0,
        "false_abstentions": 0,
    }

    # Threshold 0.35: both (0.20, 0.30 <= 0.35) generate.
    # Correct Gen = 1, Correct Abs = 0, False Accept = 1, False Abs = 0
    assert sweep[2] == {
        "threshold": 0.35,
        "correct_generations": 1,
        "correct_abstentions": 0,
        "false_accepts": 1,
        "false_abstentions": 0,
    }
