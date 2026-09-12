from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from retriever import retriever


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""
)

query = input("Ask a question about your PDF: ")

results = retriever.invoke(query)

print("\n===== RETRIEVED DOCUMENTS =====")

for i, document in enumerate(results):
    print(f"\n--- Document {i + 1} ---")
    print(document.page_content)
    print("Metadata:", document.metadata)

context = "\n\n".join(
    document.page_content for document in results
)

final_prompt = prompt.format(
    context=context,
    question=query
)

response = llm.invoke(final_prompt)

print("\n===== ANSWER =====")
print(response.content[0]["text"])

# # from dotenv import load_dotenv
# # from langchain_google_genai import ChatGoogleGenerativeAI

# # load_dotenv()

# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-3.8-flash",
# #     temperature=0
# # )

# # response = llm.invoke("What is Retrieval-Augmented Generation?")

# # print(response.content)

# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate

# from retriever import retriever


# load_dotenv()


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash",
#     temperature=0
# )


# prompt = ChatPromptTemplate.from_template(
#     """
# You are a helpful assistant answering questions based on the provided document context.

# Use ONLY the information provided in the context to answer the question.

# If the answer is not present in the context, say:
# "I couldn't find the answer in the provided document."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )


# query = input("Ask a question about your PDF: ")

# results = retriever.invoke(query)

# context = "\n\n".join(
#     document.page_content for document in results
# )

# final_prompt = prompt.format(
#     context=context,
#     question=query
# )

# response = llm.invoke(final_prompt)

# print("\nAnswer:")
# print(response.content)


