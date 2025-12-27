# DocuMind - RAG Document Intelligence System

A production-ready RAG (Retrieval Augmented Generation) system for intelligent document search and question-answering.

## Features

- **Multi-format Support**: PDF, DOCX, TXT, HTML documents
- **Semantic Search**: Uses sentence-transformers for high-quality embeddings
- **Vector Storage**: ChromaDB for efficient similarity search
- **Smart Chunking**: Overlapping chunks for better context retrieval
- **REST API**: FastAPI-powered API with automatic documentation
- **Demo UI**: Built-in web interface for testing

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000 for the demo UI.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Demo UI |
| `/health` | GET | System health check |
| `/upload` | POST | Upload a document |
| `/query` | POST | Query documents |
| `/documents` | GET | List all documents |
| `/documents/{id}` | DELETE | Delete a document |

## Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the working hours?", "top_k": 5}'
```

## Architecture

```
DocuMind
├── Document Processor  # Extract text from various formats
├── Embedding Service   # Generate semantic embeddings
├── Vector Store        # ChromaDB for similarity search
└── RAG Engine          # Orchestrate retrieval and answer synthesis
```

## Tech Stack

- FastAPI - Web framework
- Sentence Transformers - Text embeddings
- ChromaDB - Vector database
- PyPDF2 & python-docx - Document parsing

## License

MIT
