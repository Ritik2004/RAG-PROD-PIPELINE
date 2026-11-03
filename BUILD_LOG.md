# Build Log: Ask My Docs

This file explains, step by step, what was built in this project and **why**.
It is written so anyone can follow along, even without a deep AI background.

---

## What is this project?

**Ask My Docs** lets you ask questions in plain English about a set of documents (PDFs)
and get an answer **with citations** (which file and page the answer came from).

It uses **RAG (Retrieval-Augmented Generation)**:

1. **Retrieve:** find the parts of the documents that are relevant to the question.
2. **Augment:** put those parts into the prompt.
3. **Generate:** the LLM answers using only those parts.

**Why RAG?** An LLM doesn't know your private or recent documents, and it can make things up.
RAG gives it the right information at question time and makes it cite its sources.

---

## How it works (big picture)

There are two pipelines:

**1. Indexing** (runs once, when documents are added)

```
PDFs → Load → Split into chunks → Embed (turn text into vectors) → Store in a vector database
```

**2. Querying** (runs for every question)

```
Question → Search the vector database → Top matching chunks → Prompt → LLM → Answer + citations
```

---

## Tech stack

| Part | Tool | Why |
|---|---|---|
| Framework | LangChain | Ready-made, tested building blocks for each RAG step |
| PDF loading | PyPDF | Reads PDF text page by page |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits at natural boundaries (paragraphs, lines) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, fast, runs locally, 384-number vectors |
| Vector database | Qdrant (local mode) | Fast similarity search, stores metadata, open source |
| LLM | Groq (`openai/gpt-oss-20b`) via `langchain-groq` | Fast, low-cost LLM API |

(More tools are added to this table as they're used.)

---

## Steps completed

### Step 1: Load the documents

**What:** Read every PDF in `data/raw/` and turn each page into a `Document`.

A `Document` has two parts:
- `page_content`: the text of the page
- `metadata`: where it came from, e.g. `{"source": "data/raw/rag_paper.pdf", "page": 3}`

**Why:** The LLM can't read PDF files directly; we need plain text. The metadata (file + page)
is kept because it later becomes the **citation** in the answer.

**Where the data goes:** into memory (a Python list), not a database yet.
The database comes after chunking and embedding.

**Code:** [app/ingest/loader.py](app/ingest/loader.py)

**Test:** `python -m tests.test_loader` (run from the project root)

**Sample documents used for now:** three public research papers
(Attention Is All You Need, the RAG paper, ReAct agents).

**Notes for later:** with thousands of documents, loading everything at once would use too much
memory. The fix is to load and process them in **batches** (`lazy_load()` reads one page at a time).

### Step 2: Split documents into chunks

**What:** Cut each page into smaller pieces ("chunks") of up to 1000 characters,
with 200 characters of overlap between neighbouring chunks.

**Why:**
- A whole page is too big and mixes many topics. Small, focused chunks make search more accurate.
- The LLM's prompt has limited space, so we send only the few most relevant chunks.
- The **overlap** stops an important sentence from being lost when it falls on a boundary.

**How:** `RecursiveCharacterTextSplitter` tries to split at paragraph breaks first, then line
breaks, then spaces, so sentences aren't cut in half. Each chunk keeps its page's metadata
(file + page) plus `start_index` (where in the page it starts), so citations still work.

**Result:** 67 pages → 291 chunks, none longer than 1000 characters.

**Code:** [app/ingest/splitter.py](app/ingest/splitter.py)

**Test:** `python -m tests.test_splitter`

### Step 3: Embed the chunks and store them in a vector database

**What:** Turn each chunk into an **embedding** (a list of 384 numbers that captures its meaning)
and save it, with its text and metadata, in **Qdrant**, a vector database stored on disk.

**Why:** Computers can't compare meaning in plain text, but they can compare vectors.
Chunks with similar meaning get similar vectors. To answer a question, we turn the question
into a vector and ask Qdrant for the closest chunks. This works even when the question uses
different words than the document ("salary" vs "pay").

**How:** `HuggingFaceEmbeddings` (model `all-MiniLM-L6-v2`, small and free, runs locally) creates
the vectors. `QdrantVectorStore.from_documents(...)` embeds and stores all chunks in one call.
`force_recreate=True` rebuilds the collection so re-running doesn't create duplicates.

**Something we learned:** vector search **always** returns the closest chunks, even when the
question has nothing to do with the documents. Asking "Which country does Messi play for?" returned a
World Cup example from the RAG paper. So the system needs a way to say "I don't know"
(a strict prompt now, a score threshold and evals later).

**Code:** [app/ingest/indexer.py](app/ingest/indexer.py)

**Test:** `python -m tests.test_indexer`

### Step 4: Answer questions with citations (the first working RAG)

**What:** For a question, find the 4 most similar chunks, give them to the LLM along with the question,
and get back an answer that cites its sources like `[rag_paper.pdf, page 9]`.

**Why:**
- The LLM answers from **our documents**, not from memory, so it's grounded and up to date.
- **Citations** let users check the answer themselves, which builds trust.
- If the answer isn't in the documents, it says **"I don't know"** instead of making something up.

**How:**
- `format_docs` puts a label like `[rag_paper.pdf, page 9]` above each chunk.
- A **prompt template** tells the LLM: use only this context, cite the labels, otherwise say "I don't know".
- A LangChain **chain** connects the steps: `prompt → LLM (Groq) → plain text`.
- The vector database is built once (indexing) and then **loaded** for each question, not rebuilt.

**Result:**

| Question | Answer |
|---|---|
| What is retrieval-augmented generation? | A correct summary, cited `[rag_paper.pdf, page 9]` |
| How many attention heads does the Transformer use? | "8 attention heads" `[attention_is_all_you_need.pdf, page 5]` |
| Which country does Messi play for? (not in the docs) | "I don't know." |

**Code:** [app/generate/chain.py](app/generate/chain.py)

**Test:** `python -m tests.test_chain` (needs `GROQ_API_KEY`)

### Step 5: One chain for the whole pipeline

**What:** Moved retrieval inside the LangChain chain, so a single call does everything:
`question → retriever → format chunks → prompt → LLM → answer`.

**Why:** Each step is now a swappable block. In the next steps the basic retriever gets replaced
by a better one (hybrid search, then reranking) **without changing any other code**.

**How:** `store.as_retriever(k=4)` turns the vector store into a retriever. A small dict
`{"context": retriever | format_docs, "question": RunnablePassthrough()}` sends the question down
two paths at once (one fetches and formats the chunks, the other passes the question through),
and the result fills the prompt.

**Result:** same answer as before: "8 attention heads [attention_is_all_you_need.pdf, page 5]"

**Code:** `build_rag_chain` in [app/generate/chain.py](app/generate/chain.py)

**Test:** `python -m tests.test_rag_chain`

### Step 6: Measure retrieval quality (the baseline)

**What:** A small "exam" for the search step. [evals/golden_set.json](evals/golden_set.json) has 15 questions
across the 3 PDFs. For each one, the exact file and page holding the answer was checked by hand.
The script asks the retriever each question and checks whether the right page is in the top 4 chunks.

**Why:** If search doesn't find the right chunk, the LLM can't give the right answer. So retrieval is
tested on its own, without the LLM. This number is the **baseline**: every later improvement
(hybrid search, reranking) is measured against it, so "better" means a real number, not a feeling.

**Result:** **hit rate @ 4 = 7/15 (47%)**

| What we saw | What it suggests |
|---|---|
| Every miss found the **right file** but the **wrong page** | Search gets the topic, but not the precise passage |
| Misses were often on exact terms: "Adam", "BART-large", "21M documents", "MIPS", "PaLM-540B" | Meaning-based search is weak on exact names and numbers → **keyword search (BM25)** should help |
| Some answers sit in tables or number-heavy text | A **reranker** that reads the question and chunk together should pick the precise chunk |

**Code:** [evals/retrieval_eval.py](evals/retrieval_eval.py)

**Run:** `python -m evals.retrieval_eval`

### Step 7: Hybrid search (keyword + meaning)

**What:** Run two searches on every question and merge the results:
- **Vector search**: finds chunks with similar *meaning*
- **BM25 keyword search**: finds chunks containing the *exact words* (names, numbers, acronyms)

The two ranked lists are merged with **Reciprocal Rank Fusion (RRF)**: each list gives a chunk points
based on its position (`1 / (60 + rank)`), the points are added, and the chunks are re-sorted.
Chunks that **both** searches rank highly end up at the top.

**Why:** Step 6 showed that vector search found the right paper but missed exact terms like "Adam" and
"PaLM-540B". Keyword search is good at exactly those, but bad at synonyms. Together they cover each other's gaps.

**Result (same 15-question eval, top 4 chunks):**

| Retriever | Hit rate |
|---|---|
| Vector only (baseline) | 7/15 (47%) |
| BM25 only | 8/15 (53%) |
| **Hybrid (BM25 + vector, RRF)** | **9/15 (60%)** |

Hybrid beats both searches alone. The remaining misses are mostly in the RAG paper, where common words
like "RAG" and "documents" appear on almost every page, so neither search can pick the precise page.
That's the job of the next step, reranking.

**Code:** [app/retrieve/hybrid.py](app/retrieve/hybrid.py)

**Run:** `python -m evals.retrieval_eval` (set `MODE` to `"vector"` or `"hybrid"`)

---

## Coming next

- [ ] Reranking
- [ ] Evaluations (measure answer quality) + a CI check that blocks bad changes
- [ ] Tracing and monitoring (latency, cost, quality)
- [ ] Guardrails, API, UI, Docker, deployment
