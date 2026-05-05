from langchain_community.document_loaders import PyPDFLoader


def getTextFromDocuments(file_path):
    if file_path.lower().endswith('.md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


if __name__ == "__main__":
    # Test it
    text = getTextFromDocuments("bm/bm_tech_handbook.md")
    print(text[:500])  