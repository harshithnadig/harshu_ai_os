import pytest

from harshu_ai_os.rag.prompt import build_grounded_prompt


def test_build_grounded_prompt_contains_context_and_question():
    question = "How is Harshu AI OS tested?"
    context = "Harshu AI OS is tested using Pytest."

    prompt = build_grounded_prompt(question, context)

    assert context in prompt
    assert question in prompt
    assert "available context is insufficient" in prompt


def test_build_grounded_prompt_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        build_grounded_prompt(
            "   ",
            "Harshu AI OS is tested using Pytest.",
        )


def test_build_grounded_prompt_rejects_empty_context():
    with pytest.raises(
        ValueError,
        match="Context cannot be empty.",
    ):
        build_grounded_prompt(
            "How is Harshu AI OS tested?",
            "   ",
        )