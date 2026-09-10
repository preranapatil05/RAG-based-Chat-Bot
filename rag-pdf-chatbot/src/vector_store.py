from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunking import chunks


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

vector_store.save_local("faiss_index")

print("FAISS vector store created successfully.")