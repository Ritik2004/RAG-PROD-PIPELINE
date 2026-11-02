from app.ingest.indexer import load_vector_store
from app.generate.chain import answer

store = load_vector_store()

questions = [
    "What is retrieval-augmented generation?",
    "How many attention heads does the Transformer use?",
    "Which country does Messi play for?",
]
for q in questions:
    print("Q:", q)
    print("A:", answer(q, store))
    print("-" * 80)