from fastapi import FastAPI,HTTPException
from harshu_ai_os.api.schemas import AskRequest, AskResponse
from harshu_ai_os.llm.router import classify_task_with_model, choose_route
from harshu_ai_os.llm.client import call_llm
from harshu_ai_os.kernel.logger import get_logger
from harshu_ai_os.llm.exceptions import LLMServiceError


app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        classification = classify_task_with_model(request.question)

        logger = get_logger(__name__)

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