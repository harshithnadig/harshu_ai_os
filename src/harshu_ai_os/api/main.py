"""HTTP boundary for Harshu AI OS.

This file validates requests and chooses the appropriate application workflow;
retrieval and provider details remain in their owning modules.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from harshu_ai_os.api.schemas import AskRagResponse, AskRequest, AskResponse
from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.llm.router import classify_task_with_model, choose_route
from harshu_ai_os.core import get_logger
from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.service import (
    DEFAULT_MAXIMUM_DISTANCE,
    answer_with_chroma_rag,
)

app = FastAPI()
logger = get_logger(__name__)

# The standalone Vite client is only used during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Provide a dependency-free liveness check for local development."""
    return {"status": "healthy"}


def choose_request_route(question: str):
    """Classify one question and return the matching provider route."""
    classification = classify_task_with_model(question)
    return classification, choose_route(classification.complexity)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Handle the simple direct-generation path without retrieval machinery."""
    try:
        classification, route = choose_request_route(request.question)

        result = call_llm(route, request.question)

        logger.info("model=%s", route["model"])
        return {
            "complexity": classification.complexity,
            "answer": result,
            "model": route["model"],
        }
    except LLMServiceError as error:
        logger.error(error)
        raise HTTPException(
            status_code=503, detail="AI service temporarily unavailable"
        )


@app.post("/ask/rag", response_model=AskRagResponse)
def ask_rag(request: AskRequest):
    """Handle grounded answers and return the retrieval evidence to the UI."""
    try:
        classification, route = choose_request_route(request.question)

        collection = get_notes_collection()
        embedding_client = get_embedding_client()

        result = answer_with_chroma_rag(
            collection,
            embedding_client,
            request.question,
            route,
            maximum_distance=DEFAULT_MAXIMUM_DISTANCE,
        )

        logger.info(
            "rag_model=%s retrieved_ids=%s",
            route["model"],
            result["ids"],
        )

        return {
            "answer": result["answer"],
            "complexity": classification.complexity,
            "model": route["model"],
            "context": result["context"],
            "distances": result["distances"],
            "ids": result["ids"],
            "metadatas": result["metadatas"],
            "citations": result["citations"],
            "abstained": result["abstained"],
            "abstention_reason": result["abstention_reason"],
            "judge_reason": result.get("judge_reason"),
        }

    except LLMServiceError as error:
        logger.error(error)
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable",
        )
    except ValueError as error:
        # Retrieval validation errors are caused by the request or local index,
        # not an unavailable model provider.
        raise HTTPException(status_code=400, detail=str(error)) from error
