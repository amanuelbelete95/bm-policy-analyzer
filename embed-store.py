import chromadb
from text_chunker import chunkText
from Ingestion import getTextFromDocuments
import ollama

client = chromadb.Client()
collection = client.create_collection(name="pdf-collection")

chunks = chunkText(getTextFromDocuments("E:/resume folder/Amanuel Belete [cv].pdf"))

for i, chunk in enumerate(chunks):
    embeddings = ollama.embed(model='nomic-embed-text', input=chunk)
    collection.add(
        ids=[f"chunk-{i}"],
        embeddings=embeddings,
        metadatas=[{"source": "E:/resume folder/Amanuel Belete [cv].pdf"}],
        documents=[chunk]
    )

print("Chunks have been embedded and stored in the ChromaDB collection.")
# Lets print the collection to see the stored data
print(collection.get())