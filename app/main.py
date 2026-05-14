from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
from app.rag import ask_question

app = FastAPI(
    title="Restaurant SOP Bot",
    description="Internal bot for hygiene, kitchen workflow, and service SOPs",
    version="1.0.0"
)

# Allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    role: Optional[str] = "staff"    # For role-based access (Week 2)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        if len(v.strip()) < 3:
            raise ValueError("Query is too short — please ask a proper question")
        return v.strip()


# ── Response model ────────────────────────────────────────────
class QueryResponse(BaseModel):
    status: str
    query: str
    answer: str
    sources: List[str]
    role: Optional[str]


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Restaurant SOP Bot is running",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    try:
        # Get answer + docs
        answer, docs = ask_question(request.query)

        # Extract source titles
        sources = []

        for doc in docs:
            sources.append(doc.page_content[:80])

       
        return QueryResponse(
            status="success",
            query=request.query,
            answer=answer,
            sources=sources,
            role=request.role
        )

    except ValueError as e:
        # Bad input — client's fault
        raise HTTPException(status_code=422, detail=str(e))

    except FileNotFoundError as e:
        # Vector DB not built yet
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable: {str(e)}")

    except Exception as e:
        # Unexpected server error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok"}