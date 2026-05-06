"""Main query pipeline."""
from .retrieval import retrieve_documents
from .generation import generate_response


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