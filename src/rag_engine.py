from typing import List, Optional
import uuid
import time
from .models import (
    Document, SearchResult, QueryRequest, QueryResponse,
    DocumentUploadResponse, DocumentType
)
from .document_processor import DocumentProcessor, SAMPLE_DOCUMENTS
from .embeddings import EmbeddingService
from .vector_store import VectorStore


class RAGEngine:
    """Main RAG engine that orchestrates document processing and retrieval."""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.embeddings = EmbeddingService()
        self.vector_store = VectorStore()
        self.documents: dict[str, Document] = {}
        self._initialized = False

    def initialize_with_samples(self):
        """Load sample documents for demo purposes."""
        if self._initialized:
            return

        for filename, content in SAMPLE_DOCUMENTS.items():
            self.add_document(content.encode('utf-8'), filename)
        self._initialized = True

    def add_document(self, content: bytes, filename: str) -> DocumentUploadResponse:
        """Process and index a new document."""
        doc_id = str(uuid.uuid4())[:8]

        # Extract text and determine type
        text, doc_type = self.processor.extract_text(content, filename)

        # Create chunks
        chunks = self.processor.chunk_text(text)

        if not chunks:
            return DocumentUploadResponse(
                id=doc_id,
                filename=filename,
                chunks_created=0,
                message="No content could be extracted from document"
            )

        # Generate embeddings
        embeddings = self.embeddings.embed_batch(chunks)

        # Store in vector database
        self.vector_store.add_document(
            doc_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"filename": filename, "doc_type": doc_type.value}
        )

        # Keep document reference
        self.documents[doc_id] = Document(
            id=doc_id,
            filename=filename,
            doc_type=doc_type,
            content=text,
            chunks=chunks
        )

        return DocumentUploadResponse(
            id=doc_id,
            filename=filename,
            chunks_created=len(chunks),
            message=f"Document indexed successfully with {len(chunks)} chunks"
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        """Process a query and return relevant results with synthesized answer."""
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embeddings.embed_text(request.query)

        # Search vector store
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            min_score=request.min_score
        )

        # Format results
        results = []
        for r in raw_results:
            results.append(SearchResult(
                document_id=r['metadata'].get('document_id', ''),
                filename=r['metadata'].get('filename', 'Unknown'),
                chunk=r['document'],
                score=r['score'],
                metadata=r['metadata']
            ))

        # Synthesize answer from results
        answer = self._synthesize_answer(request.query, results)

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            query=request.query,
            results=results,
            answer=answer,
            processing_time_ms=round(processing_time, 2)
        )

    def _synthesize_answer(self, query: str, results: List[SearchResult]) -> str:
        """Generate a synthesized answer from search results."""
        if not results:
            return "No relevant information found in the documents."

        # Simple extractive answer - in production, use an LLM
        query_lower = query.lower()

        # Find most relevant snippet
        best_result = results[0]
        context = best_result.chunk

        # Extract key sentences that might answer the query
        sentences = context.split('.')
        relevant_sentences = []

        query_words = set(query_lower.split())
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_lower = sentence.lower()
            # Check for query word overlap
            sentence_words = set(sentence_lower.split())
            overlap = len(query_words & sentence_words)
            if overlap >= 1:
                relevant_sentences.append(sentence)

        if relevant_sentences:
            answer = '. '.join(relevant_sentences[:3]) + '.'
        else:
            # Fall back to first few sentences of best match
            answer = '. '.join(sentences[:2]).strip()
            if answer and not answer.endswith('.'):
                answer += '.'

        source_note = f"\n\n(Source: {best_result.filename})"
        return answer + source_note

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the system."""
        if doc_id in self.documents:
            self.vector_store.delete_document(doc_id)
            del self.documents[doc_id]
            return True
        return False

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "documents_count": len(self.documents),
            "total_chunks": self.vector_store.get_document_count(),
            "embeddings_loaded": self.embeddings._model is not None
        }
