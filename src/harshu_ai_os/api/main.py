from fastapi import FastAPI
from harshu_ai_os.api.schemas import AskRequest, AskResponse
from harshu_ai_os.llm.router import classify_task_with_model, choose_route
from harshu_ai_os.llm.client import call_llm

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    classification = classify_task_with_model(request.question)

    print(classification)

    route = choose_route(classification.complexity)
    print(route)

    result = call_llm(route,request.question)
   

    return {
        "question": request.question,
        "complexity": classification.complexity,
        "answer": result,
        "model": route["model"]
    }
