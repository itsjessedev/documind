from typing import List, Dict, Optional
import chromadb
import uuid


class VectorStore:
    """Vector database for storing and retrieving document embeddings."""

    def __init__(self, persist_directory: str = "./data/chroma"):
        # Use ephemeral client for demo (in-memory, no persistence needed)
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict
    ) -> int:
        """Add document chunks with embeddings to the store."""
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {**metadata, "chunk_index": i, "document_id": doc_id}
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[Dict]:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # ChromaDB returns distance, convert to similarity
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance  # Convert distance to similarity

                if score >= min_score:
                    search_results.append({
                        'id': doc_id,
                        'document': results['documents'][0][i] if results['documents'] else "",
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': score
                    })

        return search_results

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks of a document."""
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": doc_id}
        )
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return True
        return False

    def get_document_count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()

    def clear(self):
        """Clear all documents from the store."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
