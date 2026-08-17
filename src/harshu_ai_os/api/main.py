"""HTTP boundary for Harshu AI OS.

This file validates requests and exposes unified and diagnostic API endpoints;
planning, retrieval, and provider details remain in their owning modules.
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
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.llm.router import choose_route, classify_task_with_model
from harshu_ai_os.llm.tools import (
    AVAILABLE_TOOLS,
    RAG_LOOKUP_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
)
from harshu_ai_os.orchestrator import execute_request
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
    """Handle unified request orchestration (Direct, Agent, or Strict RAG)."""
    try:
        result = execute_request(request.question)
        logger.info(
            "ask_workflow=%s model=%s complexity=%s",
            result.get("workflow_used"),
            result.get("model"),
            result.get("complexity"),
        )
        return {
            "answer": result.get("answer", ""),
            "complexity": result.get("complexity", "general"),
            "workflow_used": result.get("workflow_used", "direct"),
            "model": result.get("model", ""),
            "tool_used": result.get("tool_used", False),
            "tool_calls_count": result.get("tool_calls_count", 0),
            "tool_sources": result.get("tool_sources", []),
            "citations": result.get("citations", []),
            "abstained": result.get("abstained", False),
            "abstention_reason": result.get("abstention_reason"),
            "judge_reason": result.get("judge_reason"),
            "tool_name": result.get("tool_name"),
            "tool_query": result.get("tool_query"),
            "stopped_reason": result.get("stopped_reason"),
            "steps_taken": result.get("steps_taken", 0),
        }
    except LLMServiceError as error:
        logger.error(error)
        raise HTTPException(
            status_code=503, detail="AI service temporarily unavailable"
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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
