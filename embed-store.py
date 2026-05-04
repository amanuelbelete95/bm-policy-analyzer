import chromadb
from text_chunker import chunkText
from Ingestion import getTextFromDocuments
import ollama

# from sentence_transformers import SentenceTransfromer
from sentence_transformers import SentenceTransformer

client = chromadb.Client()
# I will later insert the pdf or document for companies policies to ask some questions the model
collection = client.create_collection(name="myCv")
model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = chunkText(getTextFromDocuments("E:/resume folder/Amanuel Belete [cv].pdf"))

for i, chunk in enumerate(chunks):
    embeddings = model.encode(chunk).toList()
    print(embeddings)
    collection.add(
        ids=[f"chunk-{i}"],
        embeddings=embeddings,
        metadatas=[{"source": "E:/resume folder/Amanuel Belete [cv].pdf"}],
        documents=[chunk]
    )

print("Chunks have been embedded and stored in the ChromaDB collection.")
# Lets print the collection to see the stored data
print(collection.get())