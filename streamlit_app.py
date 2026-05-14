import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import os
import csv
from datetime import datetime

# ── Constants ─────────────────────────────────────────────────
DB_DIR = "db"
VALID_ROLES = ["All Staff", "Chef", "Cleaner", "Cashier", "Waiter"]


# ── Cached Resources (loads only once) ───────────────────────
@st.cache_resource
def load_db():
    if not os.path.exists(DB_DIR):
        st.error("Vector DB not found. Run ingest.py first.")
        st.stop()
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )


@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3")


db  = load_db()
llm = load_llm()


# ── Role Filter ───────────────────────────────────────────────
def build_role_filter(role: str) -> dict:
    if role == "All Staff":
        return {}   # No filter — sees everything
    return {
        "$or": [
            {"role": {"$eq": role}},
            {"role": {"$eq": "All Staff"}}
        ]
    }


# ── Prompt ────────────────────────────────────────────────────
def build_prompt(context: str, query: str, role: str) -> str:
    return f"""You are a Restaurant SOP Assistant for {role} staff.

Rules:
- Answer ONLY from the SOP context below
- Be clear and practical for restaurant staff
- If not found, say exactly: "Information not found in SOP."
- Do NOT make up any information

SOP Context:
{context}

Question: {query}

Answer:"""


# ── Ask Function ──────────────────────────────────────────────
def ask_question(query: str, role: str) -> dict:
    search_kwargs = {"k": 3}

    role_filter = build_role_filter(role)
    if role_filter:
        search_kwargs["filter"] = role_filter

    retriever = db.as_retriever(search_kwargs=search_kwargs)

    # ✅ Fixed: use invoke() not get_relevant_documents()
    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "Information not found in SOP.",
            "sources": []
        }

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt  = build_prompt(context, query, role)

    # ✅ Fixed: OllamaLLM returns plain string, not message object
    answer = llm.invoke(prompt)

    # ✅ Fixed: sources use real metadata, not raw chunk text
    seen    = set()
    sources = []
    for doc in docs:
        label = (
            f"{doc.metadata.get('title', 'Unknown')} "
            f"[{doc.metadata.get('role', '')}] "
            f"({doc.metadata.get('version', '')})"
        )
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return {
        "answer":  answer.strip(),
        "sources": sources
    }


# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant SOP Bot",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Restaurant SOP Bot")
st.caption("Ask anything about restaurant SOPs — hygiene, kitchen, service, and more.")


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    selected_role = st.selectbox(
        "Your Role",
        options=VALID_ROLES,
        index=0,
        help="Filters answers to SOPs relevant to your role"
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("PRJ-046 · Restaurant SOP Bot · v1.0")


# ── Chat History Init ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Render Chat History ───────────────────────────────────────
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["query"])

    with st.chat_message("assistant"):
        st.write(turn["answer"])

        if turn["sources"]:
            with st.expander("📄 Sources"):
                for src in turn["sources"]:
                    st.write(f"• {src}")


# ── Chat Input ────────────────────────────────────────────────
query = st.chat_input("Ask your SOP question...")

if query:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(query)

    # Show spinner while LLM is thinking
    with st.chat_message("assistant"):
        with st.spinner("Searching SOPs..."):
            result = ask_question(query.strip(), role=selected_role)

        st.write(result["answer"])
         # LOGGING HERE
        with open("logs/chat_logs.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                selected_role,
                query,
                result["answer"]
            ])

        if result["sources"]:
            with st.expander("📄 Sources"):
                for src in result["sources"]:
                    st.write(f"• {src}")

         # Feedback buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Helpful"):
                with open("feedback/feedback.csv", "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now(),
                        selected_role,
                        query,
                        result["answer"],
                        "Helpful"
                    ])
                st.success("Feedback saved")

        with col2:
            if st.button("👎 Not Helpful"):
                with open("feedback/feedback.csv", "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now(),
                        selected_role,
                        query,
                        result["answer"],
                        "Not Helpful"
                    ])
                st.success("Feedback saved")

    # Save to chat history
    st.session_state.chat_history.append({
        "query":   query,
        "answer":  result["answer"],
        "sources": result["sources"],
        "role":    selected_role
    })