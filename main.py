from fastapi import FastAPI
from app.rag import create_vectorstore, ask_question

app = FastAPI()

@app.on_event("startup")
def startup_event():
    create_vectorstore()

@app.get("/")
def home():
    return {"message": "Restaurant SOP Bot Running"}

@app.get("/ask")
def ask(q: str):

    results = ask_question(q)

    return {
        "query": q,
        "results": results
    }