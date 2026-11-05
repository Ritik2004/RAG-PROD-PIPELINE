from datetime import datetime
import json
from app.ingest.indexer import load_vector_store
from app.ingest.loader import load_documents
from app.ingest.splitter import split_documents
from app.retrieve.hybrid import build_hybrid_retriever
from app.retrieve.rerank import build_rerank_retriever
import time
import os

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
start = time.perf_counter()
for item in golden:
    # docs = retriever.invoke(item["question"])
    docs = retriever.invoke(item["question"])[:K]

    if is_hit(docs,item):
        hits+=1
        print(f"Hit: {item['question']}")
    else:
        print(f"Miss: {item['question']}")

elapsed = time.perf_counter() - start          # stop the timer: total seconds for all questions
print(f"Total hits: {hits}/{len(golden)}")

# build the runs result
result = {
    "date":datetime.now().isoformat(timespec="seconds"),
    "mode":MODE,
    "k":K,
    "candidates":CANDIDATES if MODE=="rerank" else None,
    "hits":hits,
    "total":len(golden),
    "seconds_per_question":round(elapsed/len(golden), 2)
}

# 2. Read the old results (or start empty on the first run)
RESULTS_FILE = "evals/results.json"
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
else:
    results = []

# add the run and save everything back to the file
results.append(result)
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

store.client.close()
