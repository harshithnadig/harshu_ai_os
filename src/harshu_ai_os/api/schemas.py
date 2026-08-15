"""Typed request and response contracts shared by FastAPI and the frontend."""

from typing import Any

from pydantic import BaseModel


class AskRequest(BaseModel):
    """The only user input accepted by both answer endpoints."""

    question: str


class WebSource(BaseModel):
    """A web reference returned by search tools."""

    title: str
    url: str


class AskResponse(BaseModel):
    """Response for a direct model answer, including any tool execution details."""

    answer: str
    complexity: str
    model: str
    tool_used: bool = False
    tool_name: str | None = None
    tool_query: str | None = None
    tool_sources: list[WebSource] = []


class Citation(BaseModel):
    """A retrieved source chunk supplied to the RAG model."""

    source: str
    chunk_id: str
    chunk_index: int | None
    distance: float


class AskRagResponse(BaseModel):
    """Response for a grounded answer, including inspectable evidence."""

    answer: str
    complexity: str
    model: str
    context: str
    distances: list[float]
    ids: list[str]
    metadatas: list[dict[str, Any]]
    citations: list[Citation]
    abstained: bool = False
    abstention_reason: str | None = None
    judge_reason: str | None = None
    retrieval_ms: float = 0.0
    judge_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
