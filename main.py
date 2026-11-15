from extract_transcipt import extract_transcript
from embeddings import create_embedding

youtube_url = input("Enter Youtube URL")

chunks = extract_transcript(youtube_url)

print("Total chunks", len(chunks))

embeddings = create_embedding()

sample_embedding = embeddings.embed_documents(
    [chunks[0].page_content]
)[0]

print("Embedding dimensions:", len(sample_embedding))
print("First 10 values:", sample_embedding[:10])
