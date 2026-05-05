from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# 1. Load embeddings + database
embeddings = OllamaEmbeddings(model="llama3")

db = Chroma(
    persist_directory="sop_db",
    embedding_function=embeddings
)

# 2. Load AI model
llm = OllamaLLM(model="llama3")

# 3. Ask function
def ask_sop(question):
    docs = db.similarity_search(question, k=5)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a strict Restaurant SOP assistant.

    RULES:
    - Use ONLY the given context
    - DO NOT summarize
    - DO NOT skip steps
    - If multiple steps exist, list ALL of them
    - Keep original order
    - If not found, say "Not in SOP"


    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)
    return response


# 4. Test
while True:
    q = input("\nAsk SOP: ")
    if q.lower() == "exit":
        break

    answer = ask_sop(q)
    print("\nBot:", answer)