from fastapi import FastAPI
from app.api.schemas import AskRequest, AskResponse, Source
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: AskRequest)-> AskResponse:
    return AskResponse(
        answer="This is a placeholder answer.",
        sources=[
            Source(file="example.pdf", page=1, text="This is a placeholder source text.")
        ],
        seconds=0.123
    )