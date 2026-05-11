# BM Policy Analyzer

A Retrieval Augmented Generation (RAG) system for querying company policy documents using semantic search and LLM-powered answers.

## Features

- **Document Loading**: Supports PDF and Markdown files
- **Text Chunking**: Configurable chunk size with overlap for better context preservation
- **Semantic Search**: Sentence transformers (all-MiniLM-L6-v2) for embeddings
- **Vector Storage**: ChromaDB for efficient similarity search
- **Query Transformation**: Automatically reformulates queries for better retrieval
- **LLM Integration**: Ollama (llama3.2) for generating contextual answers

## Requirements

- Python 3.11+
- Ollama running locally with llama3.2 model
- ChromaDB server (optional, defaults to local)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
COLLECTION_NAME=bm_policy_collection
OLLAMA_HOST=http://localhost:11434
```

4. Prepare your policy document:
   - Place your markdown or PDF file in `data/bm_tech_handbook.md`
   - Or update `DATA_FILE` in `src/config.py`

5. Create embeddings:
```bash
python embed.py
```

## Usage

Run the interactive CLI:
```bash
python main.py
```

Ask questions about company policies. Type `quit` or `exit` to stop.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `RETRIEVAL_TOP_K` | 5 | Number of documents to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `LLM_MODEL` | llama3.2 | Ollama model for generation |

## Project Structure

```
bm-policy-analyzer/
├── data/                  # Policy documents
├── src/
│   ├── chunking/         # Text chunking
│   ├── ingestion/        # Document loaders
│   ├── query/            # Retrieval & generation
│   └── storage/          # ChromaDB integration
├── main.py               # CLI entry point
├── embed.py              # Embedding generation script
└── config.py             # Configuration settings
```