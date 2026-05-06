"""ChromaDB vector store management."""
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL


_client = None
_embedding_model = None


def get_chroma_client():
    """Get or create ChromaDB client (singleton)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _client


def get_embedding_model():
    """Get or create embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_or_create_collection(name=COLLECTION_NAME):
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=name)


def create_vector_store(documents, collection_name=COLLECTION_NAME):
    """Create vector store by embedding and storing documents."""
    model = get_embedding_model()
    collection = get_or_create_collection(collection_name)

    for i, chunk in enumerate(documents):
        embedding = model.encode(chunk).tolist()
        collection.upsert(
            ids=[f"chunk-{i}"],
            embeddings=[embedding],
            documents=[chunk],
        )
        print(f"Stored chunk {i}")

    print(f"Stored {len(documents)} chunks in '{collection_name}'")
    return collection