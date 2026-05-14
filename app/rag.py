from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from typing import Optional
import os

# ---------------- CONFIG ---------------- #

DB_DIR = "db"

VALID_ROLES = {
    "Chef",
    "Cleaner",
    "Cashier",
    "Waiter",
    "All Staff"
}

# ---------------- MODELS ---------------- #

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="llama3"
)

# ---------------- LOAD VECTOR DB ---------------- #

def get_db():

    if not os.path.exists(DB_DIR):

        raise FileNotFoundError(
            "Vector DB not found. Run loader.py first."
        )

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

# ---------------- ROLE FILTER ---------------- #

def build_role_filter(role: str) -> Optional[dict]:

    if role == "All Staff":
        return None

    return {
        "$or": [
            {"role": {"$eq": role}},
            {"role": {"$eq": "All Staff"}}
        ]
    }

# ---------------- PROMPT ---------------- #

def build_prompt(
    context: str,
    query: str,
    role: str
) -> str:

    return f"""
You are a Restaurant SOP Assistant.

Answer ONLY from the SOP context.

Rules:
- Do NOT add extra explanations
- Do NOT add greetings
- Do NOT summarize
- Do NOT give suggestions
- Return only SOP information
- If answer is missing, reply exactly:
Information not found in SOP.

SOP Context:
{context}

Question:
{query}

Answer:
"""

# ---------------- SOURCE EXTRACTION ---------------- #

def extract_sources(docs):

    seen = set()
    sources = []

    for doc in docs:

        metadata = doc.metadata

        title = metadata.get(
            "title",
            "Unknown SOP"
        )

        role = metadata.get(
            "role",
            "Unknown Role"
        )

        version = metadata.get(
            "version",
            "v1"
        )

        source_label = f"{title} [{role}] ({version})"

        if source_label not in seen:

            seen.add(source_label)
            sources.append(source_label)

    return sources

# ---------------- MAIN QA FUNCTION ---------------- #

def ask_question(
    query: str,
    role: str = "All Staff"
):

    # Normalize role
    role = role.strip().title()

    # Validation
    if role not in VALID_ROLES:

        return {
            "answer": f"Invalid role '{role}'",
            "sources": []
        }

    # Load DB
    db = get_db()

    # Build filter
    role_filter = build_role_filter(role)

    search_kwargs = {
        "k": 3
    }

    if role_filter:
        search_kwargs["filter"] = role_filter

    # Retriever
    retriever = db.as_retriever(
        search_kwargs=search_kwargs
    )

    # Retrieve docs
    docs = retriever.invoke(query)

    # No docs found
    if not docs:

        return {
            "answer": "Information not found in SOP.",
            "sources": []
        }

    # Build context
    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    # Prompt
    prompt = build_prompt(
        context,
        query,
        role
    )

    # LLM response
    response = llm.invoke(prompt)

    # Extract content safely
    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    return {
        "answer": answer.strip(),
        "sources": extract_sources(docs)
    }