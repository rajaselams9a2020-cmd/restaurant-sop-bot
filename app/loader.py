from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import shutil
import os

# ---------------- CONFIG ---------------- #

DB_DIR = "db"
SOP_FILE = "data/sop.txt"

# Better embedding model for retrieval
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ---------------- LOAD + SPLIT ---------------- #

def load_and_split():

    # Check SOP file exists
    if not os.path.exists(SOP_FILE):
        raise FileNotFoundError(
            f"SOP file not found: {SOP_FILE}"
        )

    # Read SOP file
    with open(SOP_FILE, "r", encoding="utf-8") as file:
        text = file.read()

    # Empty file validation
    if not text.strip():
        raise ValueError("SOP file is empty")

    # Split SOP sections
    sections = text.split("-----------------------------------")

    documents = []

    for index, section in enumerate(sections):

        cleaned = section.strip()

        if not cleaned:
            continue

        lines = cleaned.split("\n")

        title = "Unknown SOP"
        role = "Unknown Role"

        # Extract metadata
        for line in lines:

            line = line.strip()

            if line.startswith("[DOCUMENT:"):
                title = line.replace("[DOCUMENT:", "").replace("]", "").strip()

            elif line.startswith("[ROLE:"):
                role = line.replace("[ROLE:", "").replace("]", "").strip()

            elif line.startswith("[VERSION:"):
                version = line.replace("[VERSION:", "").replace("]", "").strip()

        # Create LangChain document
        doc = Document(
            page_content=cleaned,
            metadata={
                "section_index": index,
                "title": title,
                "role": role,
                "source": SOP_FILE
            }
        )

        documents.append(doc)

    # Validation
    if not documents:
        raise ValueError(
            "No valid SOP sections found."
        )

    print(f"SECTIONS LOADED: {len(documents)}")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"TOTAL CHUNKS: {len(chunks)}")

    return chunks


# ---------------- BUILD VECTOR DB ---------------- #

def rebuild_vector_db():

    # Delete old DB
    if os.path.exists(DB_DIR):

        shutil.rmtree(DB_DIR)

        print("OLD VECTOR DB DELETED")

    # Load chunks
    chunks = load_and_split()

    # Create vector DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("VECTOR DB REBUILT SUCCESSFULLY")

    return vectorstore


# ---------------- LOAD EXISTING DB ---------------- #

def load_vector_db():

    if not os.path.exists(DB_DIR):

        raise FileNotFoundError(
            "Vector DB not found. Run loader.py first."
        )

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    return db


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    rebuild_vector_db()