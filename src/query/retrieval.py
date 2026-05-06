"""Document retrieval using vector search."""
import ollama

from src.config import LLM_MODEL, RETRIEVAL_TOP_K, RETRIEVAL_SIMILARITY_THRESHOLD, QUERY_TRANSFORM_PROMPT
from src.storage.vector_store import get_embedding_model, get_or_create_collection


def transform_query(user_query):
    """Rephrase vague queries to improve retrieval accuracy."""
    prompt = QUERY_TRANSFORM_PROMPT.format(user_query=user_query)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()


def retrieve_documents(
    query,
    top_k=RETRIEVAL_TOP_K,
    use_transform=True,
    similarity_threshold=RETRIEVAL_SIMILARITY_THRESHOLD,
):
    """Retrieve relevant document chunks using vector search."""
    if use_transform:
        transformed_query = transform_query(query)
        print(f"[Query Transformation] Original: '{query}' -> Transformed: '{transformed_query}'")
        search_query = transformed_query
    else:
        search_query = query

    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(search_query).tolist()

    collection = get_or_create_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    if similarity_threshold is not None:
        filtered_docs = []
        filtered_distances = []
        for doc, distance in zip(docs, distances):
            similarity = max(0, 1 - distance / 2)
            if similarity >= similarity_threshold:
                filtered_docs.append(doc)
                filtered_distances.append(distance)
        return filtered_docs, filtered_distances

    return docs, distances