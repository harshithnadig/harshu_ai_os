"""Typed request and response contracts shared by FastAPI and the frontend."""

from typing import Any

from pydantic import BaseModel


class AskRequest(BaseModel):
    """The only user input accepted by both answer endpoints."""

    question: str


class AskResponse(BaseModel):
    """Response for a direct model answer."""

    answer: str
    complexity: str
    model: str


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
