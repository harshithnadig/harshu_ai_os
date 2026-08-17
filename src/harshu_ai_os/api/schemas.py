"""Typed request and response contracts shared by FastAPI and the frontend."""

from typing import Any

from pydantic import BaseModel


class AskRequest(BaseModel):
    """The only user input accepted by answer endpoints."""

    question: str


class WebSource(BaseModel):
    """A web reference returned by search tools."""

    title: str
    url: str


class Citation(BaseModel):
    """A retrieved source chunk supplied to the RAG model."""

    source: str
    chunk_id: str
    chunk_index: int | None
    distance: float


class AskResponse(BaseModel):
    """Unified response for orchestrator queries across all workflows."""

    answer: str
    complexity: str
    workflow_used: str = "direct"
    model: str
    tool_used: bool = False
    tool_calls_count: int = 0
    tool_sources: list[WebSource] = []
    citations: list[Citation] = []
    abstained: bool = False
    abstention_reason: str | None = None
    judge_reason: str | None = None
    tool_name: str | None = None
    tool_query: str | None = None
    stopped_reason: str | None = None
    steps_taken: int = 0


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


class AskAgentResponse(BaseModel):
    """Response for a bounded ReAct multi-step agent query."""

    answer: str
    complexity: str
    model: str
    steps_taken: int
    tool_calls_count: int
    tool_sources: list[WebSource] = []
    stopped_reason: str
    tool_used: bool = False
