
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import MarkdownLoader
from langchain_community.document_loaders import TextLoader

#  What if the file is md or text can this function be used to get the text from the md file? Yes, you can use this function to get the text from the md file as well. You just need to change the loader to a markdown loader. For example, you can use the following code to get the text from a markdown file:

def getTextFromDocuments(file_path):
    if file_path.endswith(".md"):
        loader = MarkdownLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        loader = PyPDFLoader(file_path)
    documents = loader.load()
    text = ""
    for doc in documents:
        text += doc.page_content
    return text