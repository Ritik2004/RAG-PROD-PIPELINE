from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   


def build_vector_store(chunks,path="qdrant_db",collections="docs"):
    store = QdrantVectorStore.from_documents(
        chunks,
        embedding=get_embeddings(),
        path=path,
        collection_name=collections,
        force_recreate=True
    )
    return store

def load_vector_store(path="qdrant_db", collection="docs"):
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        path=path,
        collection_name=collection,
    )