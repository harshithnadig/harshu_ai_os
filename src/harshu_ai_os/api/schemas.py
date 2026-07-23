from pydantic import BaseModel
from typing import Any


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    complexity: str
    model: str

class Citation(BaseModel):
    source: str
    chunk_id: str
    chunk_index: int | None
    distance: float

class AskRagResponse(BaseModel):
    answer: str
    complexity: str
    model: str
    context: str
    distances: list[float]
    ids: list[str]
    metadatas: list[dict[str, Any]]
    citations: list[Citation]