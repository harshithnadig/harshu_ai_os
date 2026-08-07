import pytest
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.rag.sufficiency_judge import (
    SufficiencyVerdict,
    judge_context_sufficiency,
)


class FakeStructuredModel:
    def __init__(self, verdict):
        self.verdict = verdict

    def invoke(self, inputs):
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


class FakeChatModel:
    def __init__(self, verdict):
        self.verdict = verdict

    def with_structured_output(self, schema):
        return FakeStructuredModel(self.verdict)


def test_sufficiency_judge_returns_verdict_for_supported_question(monkeypatch):
    verdict = SufficiencyVerdict(
        answerable=True,
        reason="Context directly states ChromaDB is used.",
        supporting_chunk_ids=["note-1"],
    )

    def fake_factory(route):
        return FakeChatModel(verdict)

    monkeypatch.setattr(
        "harshu_ai_os.rag.sufficiency_judge.create_chat_model_from_route",
        fake_factory,
    )

    res = judge_context_sufficiency(
        {"model": "fake/model"},
        "What vector database is used?",
        ["ChromaDB is used in Harshu AI OS."],
        ["note-1"],
    )

    assert res.answerable is True
    assert res.supporting_chunk_ids == ["note-1"]


def test_sufficiency_judge_returns_unsupported_for_near_match_password_hashing(monkeypatch):
    verdict = SufficiencyVerdict(
        answerable=False,
        reason="Context mentions ChromaDB but does not state password hashing.",
        supporting_chunk_ids=[],
    )

    def fake_factory(route):
        return FakeChatModel(verdict)

    monkeypatch.setattr(
        "harshu_ai_os.rag.sufficiency_judge.create_chat_model_from_route",
        fake_factory,
    )

    res = judge_context_sufficiency(
        {"model": "fake/model"},
        "Does ChromaDB automatically handle user password hashing?",
        ["ChromaDB stores document embeddings."],
        ["note-1"],
    )

    assert res.answerable is False
    assert res.supporting_chunk_ids == []
    assert "password hashing" in res.reason


def test_sufficiency_judge_returns_unsupported_for_near_match_gemini_hnsw(monkeypatch):
    verdict = SufficiencyVerdict(
        answerable=False,
        reason="Context mentions Gemini Flash and HNSW separately but does not state Gemini Flash manages HNSW.",
        supporting_chunk_ids=[],
    )

    def fake_factory(route):
        return FakeChatModel(verdict)

    monkeypatch.setattr(
        "harshu_ai_os.rag.sufficiency_judge.create_chat_model_from_route",
        fake_factory,
    )

    res = judge_context_sufficiency(
        {"model": "fake/model"},
        "Does Gemini Flash manage the local HNSW vector index directly?",
        ["Simple queries go to Gemini Flash.", "HNSW is configured for cosine space."],
        ["note-1", "note-2"],
    )

    assert res.answerable is False
    assert res.supporting_chunk_ids == []


def test_sufficiency_judge_handles_malformed_output(monkeypatch):
    def fake_factory(route):
        return FakeChatModel("not a pydantic object")

    monkeypatch.setattr(
        "harshu_ai_os.rag.sufficiency_judge.create_chat_model_from_route",
        fake_factory,
    )

    with pytest.raises(LLMServiceError, match="AI service is temporarily unavailable"):
        judge_context_sufficiency(
            {"model": "fake/model"},
            "Question",
            ["Text"],
            ["note-1"],
        )


def test_sufficiency_judge_wraps_provider_failure(monkeypatch):
    def fake_factory(route):
        return FakeChatModel(RuntimeError("connection timeout"))

    monkeypatch.setattr(
        "harshu_ai_os.rag.sufficiency_judge.create_chat_model_from_route",
        fake_factory,
    )

    with pytest.raises(LLMServiceError, match="AI service is temporarily unavailable"):
        judge_context_sufficiency(
            {"model": "fake/model"},
            "Question",
            ["Text"],
            ["note-1"],
        )


def test_sufficiency_judge_raises_value_error_on_length_mismatch():
    with pytest.raises(ValueError, match="Each chunk must have one matching chunk ID"):
        judge_context_sufficiency(
            {"model": "fake/model"},
            "Question",
            ["Chunk 1", "Chunk 2"],
            ["note-1"],
        )
