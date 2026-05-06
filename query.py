import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv
import os

load_dotenv()

COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'bm_policy_collection')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def transform_query(user_query):
    """Rephrase vague queries to improve retrieval accuracy."""
    transform_prompt = f"""Given the user's question, rephrase it to be more specific and 
search-friendly for a company policy handbook Q&A system. Focus on key terms that would 
appear in official company documents. Return only the transformed query, nothing else.

User question: {user_query}
Transformed query:"""

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': transform_prompt}]
    )

    return response['message']['content'].strip()


def retrieve_documents(query, top_k=5, use_transform=True, similarity_threshold=0.5):
    """Retrieve relevant document chunks using vector search."""
    if use_transform:
        transformed_query = transform_query(query)
        print(f"[Query Transformation] Original: '{query}' -> Transformed: '{transformed_query}'")
        search_query = transformed_query
    else:
        search_query = query

    query_embedding = embedding_model.encode(search_query).tolist()

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results['documents'][0]
    distances = results['distances'][0]

    if similarity_threshold:
        filtered_docs = []
        filtered_distances = []
        for doc, distance in zip(docs, distances):
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                filtered_docs.append(doc)
                filtered_distances.append(distance)
        return filtered_docs, filtered_distances

    return docs, distances


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the 
provided context from the BM Technology Employee Handbook.

Guidelines:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I don't have enough information to answer that"
- Be concise and specific
- If you mention something from the handbook, reference it briefly
- Do not make up information
- If the question is unrelated to the handbook, politely redirect"""


def generate_response(context, user_query):
    """Generate response using LLM with context from retrieved documents."""
    user_content = f"""Context from BM Technology Handbook:
{context}

User question: {user_query}"""

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content}
        ]
    )

    return response['message']['content']


def query_documents(user_query, top_k=5, use_transform=True, similarity_threshold=0.5):
    """Main pipeline: Retrieve relevant docs and generate answer."""
    print(f"\n[Retrieval] Searching for: '{user_query}'")

    docs, distances = retrieve_documents(
        user_query,
        top_k=top_k,
        use_transform=use_transform,
        similarity_threshold=similarity_threshold
    )

    if not docs:
        return "No relevant documents found. Please try a different question.", [], []

    context = "\n\n---\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(docs)])

    print(f"[Retrieval] Retrieved {len(docs)} documents")

    for i, (doc, dist) in enumerate(zip(docs, distances)):
        similarity = (1 - dist) * 100
        print(f"  Doc {i+1}: {similarity:.1f}% similar")

    answer = generate_response(context, user_query)

    return answer, docs, distances


if __name__ == "__main__":
    print("BM Technology Handbook Q&A")
    print("=" * 40)
    print("Ask questions about company policies.\n")

    test_question = "What is the vacation policy?"
    print(f"Testing with: {test_question}\n")

    answer, docs, distances = query_documents(
        test_question,
        top_k=5,
        use_transform=False,
        similarity_threshold=0.3
    )

    print(f"\nAnswer: {answer}\n")