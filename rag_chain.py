from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template("""
     You are a helpful YouTube video assistant.

    Answer the question using ONLY the provided context.

    If the answer is not available in the context,
    say: "I could not find this information in the video."

    Context:
    {context}

    Question:
    {question}

    Answer:

    """)


    rag_chain = (
        {
        "context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
        }
        | prompt
        | llm
        |StrOutputParser()
    )

    return rag_chain