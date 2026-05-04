from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("E:/resume folder/Amanuel Belete [cv].pdf")
documents = loader.load()
print(documents[0].page_content[:200])