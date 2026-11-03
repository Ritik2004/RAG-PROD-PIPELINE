from app.ingest.indexer import load_vector_store
from app.generate.chain import build_rag_chain

store = load_vector_store()
rag_chain = build_rag_chain(store)

print(rag_chain.invoke("How many attention heads does the Transformer use?"))
