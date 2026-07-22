from pydantic import BaseModel
from typing import Any


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    complexity: str
    model: str

class AskRagResponse(BaseModel):
    answer: str
    complexity: str
    model: str
    context: str
    distances: list[float]
    ids: list[str]
    metadatas: list[dict[str, Any]]
