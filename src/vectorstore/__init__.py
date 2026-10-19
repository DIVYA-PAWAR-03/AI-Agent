"""
Vector Storage, Semantic Chunking, Embeddings, and Retrieval Package.
"""

from src.vectorstore.chroma import ChromaVectorStore, VectorStoreError
from src.vectorstore.chunker import chunk_document, split_text_recursively
from src.vectorstore.embeddings import (
    DeterministicMockEmbeddings,
    EmbeddingProvider,
    SentenceTransformerEmbeddings,
    get_embedding_provider,
)
from src.vectorstore.indexer import IndexManager, format_index_summary

__all__ = [
    "ChromaVectorStore",
    "VectorStoreError",
    "chunk_document",
    "split_text_recursively",
    "EmbeddingProvider",
    "SentenceTransformerEmbeddings",
    "DeterministicMockEmbeddings",
    "get_embedding_provider",
    "IndexManager",
    "format_index_summary",
]
