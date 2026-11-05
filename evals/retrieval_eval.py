import json
from app.ingest.indexer import load_vector_store
from app.ingest.loader import load_documents
from app.ingest.splitter import split_documents
from app.retrieve.hybrid import build_hybrid_retriever
from app.retrieve.rerank import build_rerank_retriever


K = 4
CANDIDATES = 20
with open("evals/golden_set.json") as f:
    golden = json.load(f)


store = load_vector_store()
# retriever = store.as_retriever(search_kwargs={"k":k})

MODE = "rerank"   # "vector" or "hybrid"

if MODE == "vector":
    retriever = store.as_retriever(search_kwargs={"k": K})
elif MODE == "hybrid":
    chunks = split_documents(load_documents())
    retriever = build_hybrid_retriever(chunks, store, k=K)
elif MODE == "rerank":
    chunks = split_documents(load_documents())
    retriever = build_rerank_retriever(build_hybrid_retriever(chunks, store, k=CANDIDATES), top_n=K)

def is_hit(docs,item):
    for doc in docs:
        source = doc.metadata["source"].split("/")[-1]   # "rag_paper.pdf"
        page = doc.metadata["page"] + 1                                 # hint: metadata "page"
        if source == item["expected_source"] and page in item["expected_pages"]:
            return True
    return False

hits=0
for item in golden:
    # docs = retriever.invoke(item["question"])
    docs = retriever.invoke(item["question"])[:K]

    if is_hit(docs,item):
        hits+=1
        print(f"Hit: {item['question']}")
    else:
        print(f"Miss: {item['question']}")

print(f"Total hits: {hits}/{len(golden)}")

store.client.close()
