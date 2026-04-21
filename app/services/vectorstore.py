from langchain_community.vectorstores import Chroma
from app.services.embeddings import embeddings

def get_vectorstore():
    return Chroma(
        collection_name="rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )