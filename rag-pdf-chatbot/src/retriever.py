from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

query = input("Ask a question: ")

results = retriever.invoke(query)

for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print("Metadata:", result.metadata)