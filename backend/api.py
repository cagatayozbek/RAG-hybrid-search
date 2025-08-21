from fastapi import FastAPI
from rerank_pipeline import ask_query

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(query: str):
    return {"answer": ask_query(query)}