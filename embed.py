"""Create embeddings and store in vector database."""
from src.config import DATA_FILE, COLLECTION_NAME
from src.ingestion import get_text_from_documents
from src.chunking import chunk_text
from src.storage import create_vector_store


def main():
    print(f"Loading document: {DATA_FILE}")
    text = get_text_from_documents(str(DATA_FILE))
    print(f"Document loaded ({len(text)} characters)")

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print("Done!")


if __name__ == "__main__":
    main()