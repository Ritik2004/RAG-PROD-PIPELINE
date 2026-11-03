from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def build_hybrid_retriever(chunks, store, k=4):
    bm25 = BM25Retriever.from_documents(chunks)    # hint: what does BM25 search over?
    bm25.k = k

    vector = store.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])
