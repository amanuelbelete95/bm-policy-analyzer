"""Document loaders for various file formats."""
from langchain_community.document_loaders import PyPDFLoader


def get_text_from_documents(file_path):
    """Load and extract text from documents (PDF or Markdown)."""
    if file_path.lower().endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


if __name__ == "__main__":
    from src.config import DATA_FILE

    text = get_text_from_documents(str(DATA_FILE))
    print(text[:500])