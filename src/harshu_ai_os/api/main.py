from harshu_ai_os.api.schemas import AskRagResponse
from fastapi import FastAPI,HTTPException
from harshu_ai_os.llm.router import classify_task_with_model, choose_route
from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.kernel.logger import get_logger
from harshu_ai_os.llm.exceptions import LLMServiceError
from harshu_ai_os.api.schemas import (
    AskRequest,
    AskResponse,
    AskRagResponse,
)
from harshu_ai_os.rag.chroma_store import get_notes_collection
from harshu_ai_os.rag.embedding_client import get_embedding_client
from harshu_ai_os.rag.generator import create_rag_generator
from harshu_ai_os.rag.service import answer_with_chroma_rag

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    logger = get_logger(__name__)
    try:
        classification = classify_task_with_model(request.question)

        

        logger.info(classification)

        route = choose_route(classification.complexity)

        logger.info(route)

        result = call_llm(route, request.question)

        logger.info("model=%s", route["model"])
        return {
            "complexity": classification.complexity,
            "answer": result,
            "model": route["model"]
        }
    except LLMServiceError as error:
        logger.error(error)
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable"
        )

@app.post("/ask/rag", response_model=AskRagResponse)
def ask_rag(request: AskRequest):
    logger = get_logger(__name__)

    try:
        classification = classify_task_with_model(request.question)
        route = choose_route(classification.complexity)

        collection = get_notes_collection()
        embedding_client = get_embedding_client()
        generate_text = create_rag_generator(route)

        result = answer_with_chroma_rag(
            collection,
            embedding_client,
            request.question,
            generate_text,
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
            "citations": result["citations"]
        }

    except LLMServiceError as error:
        logger.error(error)
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable",
        )