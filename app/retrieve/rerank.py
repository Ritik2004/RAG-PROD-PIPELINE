from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


def build_rerank_retriever(base_retriever, top_n=4):
    # TODO 1: load the cross-encoder model (same as in tests/try_rerank_real.py)
    reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # TODO 2: create a CrossEncoderReranker with that model, keeping top_n chunks
    compressor = CrossEncoderReranker(model=reranker, top_n=top_n)

    # TODO 3: return a ContextualCompressionRetriever that runs base_retriever first,
    #         then the reranker on its results
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )
