from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(folder="data/raw"):
    loader = PyPDFDirectoryLoader(folder)
    docs = loader.load()
    return docs         