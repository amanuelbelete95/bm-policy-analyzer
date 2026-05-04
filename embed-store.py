import chromadb
from text_chunker import chunkText
from Ingestion import getTextFromDocuments
from sentence_transformers import SentenceTransformer


PDF_PATH = "E:/resume folder/Amanuel Belete [cv].pdf"
COLLECTION_NAME = "bm_policy_analyzer"
def create_vector_store():
    # Load and chunk the document
    text = getTextFromDocuments(PDF_PATH)
    chunks = chunkText(text)

      # Create ChromaDB client (persistent)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Embeding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Embed and store chunks in ChromaDB
    for i, chunk in enumerate(chunks):
    embeddings = model.encode(chunk).tolist()
    print(embeddings)
    collection.add(
        ids=[f"chunk-{i}"],
        embeddings=embeddings,
        metadatas=[{"source": PDF_PATH,}],
        documents=[chunk]
    )
    print(f"Stored {len(chunks)} chunks in '{COLLECTION_NAME}'")
    return collection