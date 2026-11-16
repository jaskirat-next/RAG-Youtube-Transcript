from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def create_llm():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

    return model