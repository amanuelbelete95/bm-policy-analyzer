"""LLM response generation."""
import ollama

from src.config import LLM_MODEL, SYSTEM_PROMPT


def generate_response(context, user_query):
    """Generate response using LLM with context from retrieved documents."""
    user_content = f"""Context from BM Technology Handbook:

{context}

User question: {user_query}"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
    )

    return response["message"]["content"]