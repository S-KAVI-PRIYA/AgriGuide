from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import backend

app = FastAPI(title="Agri Guide API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

collection = backend.get_or_build_collection()


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok", "records": collection.count()}


@app.post("/api/ask")
def ask(payload: AskRequest):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = backend.answer_question(collection, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": answer}