from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.ingest.indexer import load_vector_store
from app.ingest.loader import load_documents
from app.ingest.splitter import split_documents
from app.retrieve.hybrid import build_hybrid_retriever

QUESTION = "How many attention heads does the Transformer use?"   # expected: attention_is_all_you_need.pdf, page 5

# TODO 1: build the hybrid retriever with k=10 (copy how evals/retrieval_eval.py does it)
store = load_vector_store()
chunks = split_documents(load_documents())
retriever = build_hybrid_retriever(chunks, store, k=10)

# TODO 2: get the candidate chunks for QUESTION
retrieved_docs = retriever.invoke(QUESTION)

# TODO 3: load the reranker and score every (QUESTION, chunk text) pair
reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.score([(QUESTION, doc.page_content) for doc in retrieved_docs])

# TODO 4: print each chunk sorted by score (high to low):
#         search rank, score, file name, page, first 80 characters of the text
results = [(rank, doc, score) for rank, (doc, score) in enumerate(zip(retrieved_docs, scores), start=1)]
results = sorted(results, key=lambda x: x[2], reverse=True)


for rank, doc, score in results:
    source = doc.metadata["source"].split("/")[-1]
    page = doc.metadata["page"] + 1
    text = doc.page_content[:80].replace("\n", " ")
    print(f"search rank {rank:>2} | score {score:6.2f} | {source} p{page} | {text}")

# TODO 5: close the store
store.client.close()
