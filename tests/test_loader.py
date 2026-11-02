from app.ingest.loader import load_documents

docs = load_documents()

# 1. print the total number of pages loaded
print("Total pages:", len(docs))

# 2. print the metadata of the first doc
print("Metadata:", docs[0].metadata)

# 3. print the first 300 characters of the first doc
print("Text:", docs[0].page_content[:300])

counts = {}
for doc in docs:
    source = doc.metadata["source"]
    counts[source] = counts.get(source, 0) + 1
print(counts)