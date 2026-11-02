from app.ingest.loader import load_documents
from app.ingest.splitter import split_documents
from app.ingest.indexer import build_vector_store

chunks = split_documents(load_documents())
store = build_vector_store(chunks)
print("Vector store built with", len(chunks), "chunks.")

question = "which country messi plays for?"
results = store.similarity_search(question, k=3)
# print("Top 3 results for the question:", results)

for doc in results:
    print("Source:", doc.metadata["source"])
    print("Text:", doc.page_content[:300])  # Print first 300 characters of the chunk
    print("-" * 80)