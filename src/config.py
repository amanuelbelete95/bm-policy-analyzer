"""Centralized configuration for BM Policy Analyzer."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "bm_tech_handbook.md"

CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "bm_policy_collection")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

RETRIEVAL_TOP_K = 5
RETRIEVAL_SIMILARITY_THRESHOLD = 0.2

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the
provided context from the BM Technology Employee Handbook.

Guidelines:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I don't have enough information to answer that"
- Be concise and specific
- If you mention something from the handbook, reference it briefly
- Do not make up information
- If the question is unrelated to the handbook, politely redirect"""

QUERY_TRANSFORM_PROMPT = """Given the user's question, rephrase it to be more specific and
search-friendly for a company policy handbook Q&A system. Focus on key terms that would
appear in official company documents. Return only the transformed query, nothing else.

User question: {user_query}
Transformed query:"""