from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"


class Document(BaseModel):
    id: str
    filename: str
    doc_type: DocumentType
    content: str
    chunks: List[str] = []
    metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class SearchResult(BaseModel):
    document_id: str
    filename: str
    chunk: str
    score: float
    metadata: dict = {}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.3


class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    answer: str
    processing_time_ms: float


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    documents_count: int
    embeddings_loaded: bool
