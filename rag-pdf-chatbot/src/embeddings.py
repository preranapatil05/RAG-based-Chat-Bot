from langchain_huggingface import HuggingFaceEmbeddings
from chunking import chunks


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(f"Number of chunks: {len(chunks)}")
print(f"Embedding dimensions: {len(vectors[0])}")