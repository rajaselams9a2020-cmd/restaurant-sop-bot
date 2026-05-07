from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.loader import load_and_split
import os

DB_DIR = "db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

def create_vectorstore():

    if os.path.exists(DB_DIR):
        print("VECTOR DB ALREADY EXISTS")
        return

    documents = load_and_split()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("VECTORSTORE CREATED")

def ask_question(query):

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    results = vectorstore.similarity_search(query, k=3)

    print("RESULTS:", results)

    return [doc.page_content for doc in results]