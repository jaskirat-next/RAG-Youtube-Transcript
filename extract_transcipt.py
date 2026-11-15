from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_transcript(youtube_url):

    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        language=["en"]
    )

    documents = loader.load()


    # text splitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return chunks