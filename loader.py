from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split():
    loader = TextLoader("data/sop.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(documents)

    print("TOTAL CHUNKS:", len(chunks))

    for chunk in chunks:
        print("CHUNK:")
        print(chunk.page_content)
        print("----------------")

    return chunks