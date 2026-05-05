import chromadb
from text_chunker import chunkText
from Ingestion import getTextFromDocuments
from sentence_transformers import SentenceTransformer


PDF_PATH = "bm/bm_tech_handbook.md"
COLLECTION_NAME = "bm_policy_collection"

def create_vector_store():
    # Load and chunk the document
    text = getTextFromDocuments(PDF_PATH)
    chunks = chunkText(text)

    # Create ChromaDB client (persistent)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed and store chunks in ChromaDB
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        
        # Store the chunk and its embedding in ChromaDB
        collection.upsert(
            ids=[f"chunk-{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": PDF_PATH}]
        )
        print(f"Stored chunk {i}")

    print(f"Stored {len(chunks)} chunks in '{COLLECTION_NAME}'")
    return collection


if __name__ == "__main__":
    create_vector_store()