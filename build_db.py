from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Load SOP file
loader = TextLoader("sop.txt")
documents = loader.load()

# 2. Split text into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(documents)

# 3. Convert text → embeddings
embeddings = OllamaEmbeddings(model="llama3")

# 4. Store in vector DB
db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="sop_db"
)

db.persist()

print("SOP Database created successfully")