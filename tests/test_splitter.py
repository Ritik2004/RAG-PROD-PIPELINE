from app.ingest.loader import load_documents
from app.ingest.splitter import split_documents

docs = load_documents()
split_docs = split_documents(docs)

print("Pages:", len(docs), "Chunks:", len(split_docs))

# 2. print the metadata of chunk 5 (does it still have source + page?)
print("Metadata:", split_docs[8].metadata)

# 3. print the length of chunk 5's text
print("Length:", len(split_docs[8].page_content))

# 4. find the biggest chunk length (hint: max() + list comprehension)
print("Max length:", max(len(chunk.page_content) for chunk in split_docs))