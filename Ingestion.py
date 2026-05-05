from langchain_community.document_loaders import PyPDFLoader


def getTextFromDocuments(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


if __name__ == "__main__":
    # Test it
    text = getTextFromDocuments("E:/resume folder/Amanuel Belete [cv].pdf")
    print(text[:200])x  