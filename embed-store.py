import chromadb
from text_chunker import chunkText
from Ingestion import getTextFromDocuments
import ollama

client = chromadb.Client()
collection = client.create_collection(name="pdf-collection")

chunks = chunkText(getTextFromDocuments("E:/resume folder/Amanuel Belete [cv].pdf"))

embeddings = ollama.embed(model='nomic-embed-text', input=chunks)

print(embeddings)