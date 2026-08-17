"""HTTP boundary for Harshu AI OS.

This file validates requests and chooses the appropriate application workflow;
retrieval and provider details remain in their owning modules.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from harshu_ai_os.agents.loop import run_agent_loop
from harshu_ai_os.api.schemas import (
    AskAgentResponse,
    AskRagResponse,
    AskRequest,
    AskResponse,
)
from harshu_ai_os.core import get_logger
from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.llm.router import choose_route, classify_task_with_model
from harshu_ai_os.llm.tools import (
    AVAILABLE_TOOLS,
    RAG_LOOKUP_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
)
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
    """Handle the direct-generation path with optional single-round tool calling."""
    try:
        classification, route = choose_request_route(request.question)

        result = call_llm(
            route,
            request.question,
            tools=[WEB_SEARCH_TOOL_SCHEMA],
            available_tools=AVAILABLE_TOOLS,
            return_tool_info=True,
        )

        logger.info("model=%s", route["model"])
        if isinstance(result, dict):
            return {
                "complexity": classification.complexity,
                "answer": result.get("answer", ""),
                "model": route["model"],
                "tool_used": result.get("tool_used", False),
                "tool_name": result.get("tool_name"),
                "tool_query": result.get("tool_query"),
                "tool_sources": result.get("tool_sources", []),
            }

        return {
            "complexity": classification.complexity,
            "answer": str(result),
            "model": route["model"],
            "tool_used": False,
            "tool_name": None,
            "tool_query": None,
            "tool_sources": [],
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
            "retrieval_ms": result.get("retrieval_ms", 0.0),
            "judge_ms": result.get("judge_ms", 0.0),
            "generation_ms": result.get("generation_ms", 0.0),
            "total_ms": result.get("total_ms", 0.0),
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


@app.post("/ask/agent", response_model=AskAgentResponse)
def ask_agent(request: AskRequest):
    """Handle bounded ReAct multi-step agent queries."""
    try:
        classification, route = choose_request_route(request.question)

        result = run_agent_loop(
            route=route,
            user_prompt=request.question,
            tools=[WEB_SEARCH_TOOL_SCHEMA, RAG_LOOKUP_TOOL_SCHEMA],
            available_tools=AVAILABLE_TOOLS,
        )

        logger.info(
            "agent_model=%s steps=%s",
            route["model"],
            result.get("steps_taken", 0),
        )

        return {
            "answer": result.get("answer", ""),
            "complexity": classification.complexity,
            "model": route["model"],
            "steps_taken": result.get("steps_taken", 0),
            "tool_calls_count": result.get("tool_calls_count", 0),
            "tool_sources": result.get("tool_sources", []),
            "stopped_reason": result.get("stopped_reason", "completed"),
            "tool_used": result.get("tool_used", False),
        }
    except LLMServiceError as error:
        logger.error(error)
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable",
        )

