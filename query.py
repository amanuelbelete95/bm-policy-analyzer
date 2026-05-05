import chromadb
from sentence_transformers import SentenceTransformer
# CAN WE USE THE    OLLAMA API KEY IN THIS CODE? YES, BUT WE NEED TO IMPORT THE OLLAMA LIBRARY FIRST.
import ollama


# How can Import env variables in python?
from dotenv import load_dotenv
load_dotenv()
# after loading env variables, you can access them using os.environ
import os
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY')
COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'my_collection') 



def query_documents(user_query, top_k=3):
    # 1. Embed user query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(user_query).tolist()

    # 2. Load collection and query
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 3. Get relevant chunks
    relevant_chunks = results['documents'][0]
    context = "\n".join(relevant_chunks)

    # 4. Generate response with Ollama
    prompt = f"Context from document:\n{context}\n\nUser question: {user_query}\n\nProvide a helpful answer based on the context."


#    HOW CAN WE USE THE OLLAMA API KEY IN THIS CODE? WE CAN SET THE API KEY IN THE OLLAMA CLIENT CONFIGURATION, OR IF THE OLLAMA LIBRARY AUTOMATICALLY READS FROM ENV VARIABLES, THEN IT WILL USE THE OLLAMA_API_KEY WE SET EARLIER.

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']


if __name__ == "__main__":
    print("Document Q&A - Type 'exit' to quit\n")
    while True:
        user_input = input("Your question: ")
        if user_input.lower() == 'exit':
            break
        answer = query_documents(user_input)
        print(f"\nAnswer: {answer}\n")