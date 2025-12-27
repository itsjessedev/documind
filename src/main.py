from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from .models import QueryRequest, QueryResponse, DocumentUploadResponse, HealthResponse
from .rag_engine import RAGEngine


# Global RAG engine instance
rag_engine: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engine on startup."""
    global rag_engine
    rag_engine = RAGEngine()
    rag_engine.initialize_with_samples()
    yield
    # Cleanup on shutdown
    rag_engine = None


app = FastAPI(
    title="DocuMind",
    description="RAG-powered Document Intelligence System - Search and query your documents using AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve demo UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DocuMind - Document Intelligence</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0A0A0F 0%, #1A1A2E 100%);
                min-height: 100vh;
                color: #E5E7EB;
            }
            .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
            header {
                text-align: center;
                margin-bottom: 40px;
            }
            h1 {
                font-size: 2.5rem;
                background: linear-gradient(135deg, #6366F1, #8B5CF6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            .subtitle { color: #9CA3AF; font-size: 1.1rem; }
            .search-box {
                background: #1F2937;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
            }
            .search-input {
                width: 100%;
                padding: 16px 20px;
                border: 2px solid #374151;
                border-radius: 12px;
                background: #111827;
                color: white;
                font-size: 1rem;
                outline: none;
                transition: border-color 0.2s;
            }
            .search-input:focus { border-color: #6366F1; }
            .search-input::placeholder { color: #6B7280; }
            .btn {
                background: linear-gradient(135deg, #6366F1, #8B5CF6);
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                margin-top: 16px;
                transition: transform 0.2s, opacity 0.2s;
            }
            .btn:hover { transform: translateY(-2px); opacity: 0.9; }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
            .results {
                background: #1F2937;
                border-radius: 16px;
                padding: 24px;
                display: none;
            }
            .results.active { display: block; }
            .answer {
                background: linear-gradient(135deg, #6366F120, #8B5CF620);
                border-left: 4px solid #6366F1;
                padding: 20px;
                border-radius: 0 12px 12px 0;
                margin-bottom: 24px;
            }
            .answer h3 { color: #6366F1; margin-bottom: 10px; }
            .answer p { line-height: 1.6; }
            .sources h3 { margin-bottom: 16px; color: #9CA3AF; }
            .source-item {
                background: #111827;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 12px;
            }
            .source-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            .source-file { color: #6366F1; font-weight: 500; }
            .source-score {
                background: #374151;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 0.85rem;
            }
            .source-text { color: #9CA3AF; font-size: 0.9rem; line-height: 1.5; }
            .stats {
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-top: 40px;
            }
            .stat {
                background: #1F2937;
                padding: 16px 24px;
                border-radius: 12px;
                text-align: center;
            }
            .stat-value { font-size: 1.5rem; font-weight: 700; color: #6366F1; }
            .stat-label { font-size: 0.85rem; color: #6B7280; }
            .examples {
                margin-top: 16px;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }
            .example {
                background: #374151;
                padding: 8px 14px;
                border-radius: 20px;
                font-size: 0.85rem;
                cursor: pointer;
                transition: background 0.2s;
            }
            .example:hover { background: #4B5563; }
            .loading {
                display: none;
                text-align: center;
                padding: 40px;
            }
            .loading.active { display: block; }
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #374151;
                border-top-color: #6366F1;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            .time { color: #6B7280; font-size: 0.85rem; margin-top: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>DocuMind</h1>
                <p class="subtitle">RAG-Powered Document Intelligence</p>
            </header>

            <div class="search-box">
                <input type="text" class="search-input" id="query" placeholder="Ask a question about your documents...">
                <div class="examples">
                    <span class="example" onclick="setQuery('What are the working hours?')">Working hours</span>
                    <span class="example" onclick="setQuery('How does version history work?')">Version history</span>
                    <span class="example" onclick="setQuery('What is the rate limit for the API?')">API rate limits</span>
                    <span class="example" onclick="setQuery('What is the vacation policy?')">Vacation policy</span>
                </div>
                <button class="btn" id="searchBtn" onclick="search()">Search Documents</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Searching documents...</p>
            </div>

            <div class="results" id="results">
                <div class="answer">
                    <h3>Answer</h3>
                    <p id="answerText"></p>
                    <p class="time" id="timeText"></p>
                </div>
                <div class="sources">
                    <h3>Relevant Sources</h3>
                    <div id="sourcesList"></div>
                </div>
            </div>

            <div class="stats" id="stats"></div>
        </div>

        <script>
            async function search() {
                const query = document.getElementById('query').value.trim();
                if (!query) return;

                document.getElementById('results').classList.remove('active');
                document.getElementById('loading').classList.add('active');
                document.getElementById('searchBtn').disabled = true;

                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, top_k: 5, min_score: 0.3 })
                    });
                    const data = await response.json();

                    document.getElementById('answerText').textContent = data.answer;
                    document.getElementById('timeText').textContent = `Processed in ${data.processing_time_ms}ms`;

                    const sourcesList = document.getElementById('sourcesList');
                    sourcesList.innerHTML = data.results.map(r => `
                        <div class="source-item">
                            <div class="source-header">
                                <span class="source-file">${r.filename}</span>
                                <span class="source-score">${(r.score * 100).toFixed(0)}% match</span>
                            </div>
                            <p class="source-text">${r.chunk.substring(0, 200)}...</p>
                        </div>
                    `).join('');

                    document.getElementById('results').classList.add('active');
                } catch (error) {
                    console.error('Search failed:', error);
                } finally {
                    document.getElementById('loading').classList.remove('active');
                    document.getElementById('searchBtn').disabled = false;
                }
            }

            function setQuery(q) {
                document.getElementById('query').value = q;
            }

            document.getElementById('query').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') search();
            });

            async function loadStats() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('stats').innerHTML = `
                        <div class="stat">
                            <div class="stat-value">${data.documents_count}</div>
                            <div class="stat-label">Documents</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${data.embeddings_loaded ? 'Ready' : 'Loading'}</div>
                            <div class="stat-label">AI Status</div>
                        </div>
                    `;
                } catch (error) {
                    console.error('Failed to load stats:', error);
                }
            }
            loadStats();
        </script>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and statistics."""
    stats = rag_engine.get_stats()
    return HealthResponse(
        status="healthy",
        documents_count=stats["documents_count"],
        embeddings_loaded=stats["embeddings_loaded"]
    )


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a new document."""
    allowed_extensions = {'pdf', 'docx', 'txt', 'html', 'htm'}
    ext = file.filename.lower().split('.')[-1]

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")

    try:
        result = rag_engine.add_document(content, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using natural language."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = rag_engine.query(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the system."""
    if rag_engine.delete_document(doc_id):
        return {"message": f"Document {doc_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Document not found")


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    return {
        "documents": [
            {"id": doc.id, "filename": doc.filename, "chunks": len(doc.chunks)}
            for doc in rag_engine.documents.values()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
