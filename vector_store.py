from langchain_chroma import Chroma

def create_vectore_store(chunks, embeddings):
    vectore_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="youtube_transcript3",
        persist_directory="./chroma_db"
    )

    return vectore_store