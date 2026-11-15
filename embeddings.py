from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

def create_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "gemini-embedding-001"
    )

    return embeddings
