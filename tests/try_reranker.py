from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_reranke():
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")   


result = get_reranke().score([("What is RAG?", "I used an old rag to clean the kitchen table."),
("What is RAG?", "This paper does not discuss RAG.")
])

print(result)