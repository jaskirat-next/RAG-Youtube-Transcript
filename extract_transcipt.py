from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

youtube_url = input("Enter Youtube URL: ")

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

print("Total documents: ", len(documents))
print("Total Chunks:", chunks)


for index, chunk in enumerate(chunks[:3]):
    print(f"\nDocument {index + 1}")
    print("Length:", len(chunk.page_content))
    print("Metadata:", chunk.metadata)
    print("Content:", chunk.page_content[:1000])