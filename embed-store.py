import chromadb
from text_chunker import chunkText
from Ingestion import getTextFromDocuments
from sentence_transformers import SentenceTransformer

# What is this client? The client is the connection to the ChromaDB database. It allows us to create collections, add data to the collections, and query the collections. 
# We will use the client to create a collection called office_policies and store the chunks of the pdf file and their embeddings in that collection.
client = chromadb.Client()
# I want to create a collection called office_policies to store the chunks of the pdf file and their embeddings.
collection = client.create_collection(name="office_policies");
model = SentenceTransformer('all-MiniLM-L6-v2')
# how can I pass to my pdf in the getTextFromDocuments function? You can pass the file path of your pdf file as an argument to the getTextFromDocuments function. For example, if your pdf file is located at "E:/resume folder/Amanuel Belete [cv].pdf", you can call the function like this: getTextFromDocuments("E:/resume folder/Amanuel Belete [cv].pdf"). This will return the text content of the pdf file, which you can then pass to the chunkText function to get the chunks of the text for embedding and storing in the ChromaDB collection.
chunks = chunkText(getTextFromDocuments("E:/resume folder/Amanuel Belete [cv].pdf"))


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