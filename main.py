from extract_transcipt import extract_transcript
from embeddings import create_embedding
from vector_store import create_vectore_store
from llm import create_llm
from rag_chain import create_rag_chain

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

llm = create_llm()

rag_chain = create_rag_chain(retriever, llm)

while True:
    question = input("\nAsk Your Qurstion (type exit to stop)")

    if question.lower() == "exit":
        break

    response = rag_chain.invoke(question)
    print("\nFinal Answer:")
    print(response)