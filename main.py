from extract_transcipt import extract_transcript
from embeddings import create_embedding
from vector_store import create_vectore_store
from llm import create_llm

youtube_url = input("Enter Youtube URL")

chunks = extract_transcript(youtube_url)

print("Total chunks", len(chunks))

embeddings = create_embedding()

vector_store = create_vectore_store(
    chunks,
    embeddings
)


print("Vector Store created successfully!")

print("Total documents stored:", vector_store._collection.count())


retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)

result = retriever.invoke(
    "what is this video about ?"
)


context = "\n\n".join(
    doc.page_content for doc in result
)

llm = create_llm()

question = input("Ask your question: ")

prompt = f"""
You are a helpful assistant.

Answer the question using only the given context.
If the answer is not available in the context, say:
"I could not find this information in the video."

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)


print("\nFinal Answer:")
print(response.content)