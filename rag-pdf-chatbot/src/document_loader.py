from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "data/sample.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Number of pages: {len(documents)}")

for document in documents[:2]:
    print("\n--- Page ---")
    print(document.page_content[:1000])
    print("Metadata:", document.metadata)