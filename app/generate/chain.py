from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROMPT = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below.
After each fact, cite its source exactly like [file, page X].
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
""")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
chain = PROMPT | llm | StrOutputParser()


def format_docs(docs):
    parts = []
    for doc in docs:
        source = doc.metadata["source"].split("/")[-1]   # "rag_paper.pdf"
        page = doc.metadata["page"] + 1                                 # hint: metadata "page" starts at 0, so add 1
        parts.append(f"[{source}, page {page}]\n{doc.page_content}")  # hint: the chunk's text
    return "\n\n".join(parts)


def answer(question, store, k=4):
    docs = store.similarity_search(question, k=k)
    return chain.invoke({"context": format_docs(docs), "question": question})