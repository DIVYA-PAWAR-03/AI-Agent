"""
ChromaDB persistent vector store with cosine similarity, metadata filtering, and thresholding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.config import Settings

from src.config import config
from src.models.schemas import RetrievedPassage, TextChunk
from src.utils.logger import setup_logger
from src.vectorstore.embeddings import EmbeddingProvider, get_embedding_provider

logger = setup_logger("chroma_store")


class VectorStoreError(Exception):
    """Base exception for vector store operations."""


class ChromaVectorStore:
    """
    Manages persistent ChromaDB vector storage, chunk upserts, and filtered semantic retrieval.
    """

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        collection_name: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.persist_dir = (
            Path(persist_dir).resolve()
            if persist_dir
            else config.chroma_persist_dir.resolve()
        )
        self.collection_name = collection_name or config.chroma_collection_name
        self.embedding_provider = embedding_provider or get_embedding_provider()

        self._ensure_persist_dir()
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._get_or_create_collection()

    def _ensure_persist_dir(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _get_or_create_collection(self):
        """Retrieves or initializes collection with cosine space distance."""
        try:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.error("Failed to initialize Chroma collection '%s': %s", self.collection_name, exc)
            raise VectorStoreError(f"ChromaDB initialization error: {exc}") from exc

    def count(self) -> int:
        """Returns total number of chunks currently stored in the collection."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def add_chunks(self, chunks: List[TextChunk], batch_size: int = 100) -> int:
        """
        Upserts a list of TextChunk objects with embeddings and provenance metadata.
        Uses batching to handle large document sets efficiently.
        """
        if not chunks:
            return 0

        total_added = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]

            # Generate batch embeddings
            embeddings = self.embedding_provider.embed_texts(texts)

            # Sanitize metadata for ChromaDB (must be str, int, float, or bool)
            metadatas: List[Dict[str, Any]] = []
            for c in batch:
                meta = {
                    "doc_id": str(c.doc_id),
                    "filename": str(c.filename),
                    "page_number": int(c.page_number),
                    "chunk_index": int(c.chunk_index),
                    "chunk_id": str(c.chunk_id),
                    "char_count": int(c.char_count),
                    "word_count": int(c.word_count),
                }
                # Include extra metadata if primitive type
                for k, v in c.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                metadatas.append(meta)

            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_added += len(batch)

        logger.debug("Added/updated %d chunks in collection '%s'", total_added, self.collection_name)
        return total_added

    def delete_document(self, doc_id: str) -> None:
        """Removes all indexed chunks associated with a specific document ID."""
        try:
            self._collection.delete(where={"doc_id": str(doc_id)})
            logger.debug("Deleted chunks for doc_id: %s", doc_id)
        except Exception as exc:
            logger.warning("Error deleting chunks for doc_id %s: %s", doc_id, exc)

    def get_indexed_doc_ids(self) -> Set[str]:
        """Returns set of unique doc_ids present in the collection."""
        try:
            results = self._collection.get(include=["metadatas"])
            metas = results.get("metadatas") or []
            return {m["doc_id"] for m in metas if m and "doc_id" in m}
        except Exception as exc:
            logger.warning("Could not fetch indexed doc_ids: %s", exc)
            return set()

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievedPassage]:
        """
        Executes semantic search against the vector store.
        - query_text: User search phrase.
        - top_k: Maximum number of passages to return.
        - doc_ids: Optional list of document IDs to restrict the search.
        - min_score: Threshold below which passages are excluded.
        """
        if not query_text or not query_text.strip():
            logger.debug("Empty query passed to vector store; returning 0 results.")
            return []

        k = top_k if top_k is not None and top_k > 0 else config.retrieval_top_k
        threshold = min_score if min_score is not None else config.retrieval_min_score

        # Check if collection is empty
        total_items = self.count()
        if total_items == 0:
            logger.warning("Vector store collection '%s' is empty.", self.collection_name)
            return []

        # Build ChromaDB metadata filter
        where_filter: Optional[Dict[str, Any]] = None
        if doc_ids:
            clean_ids = [str(d) for d in doc_ids if d]
            if len(clean_ids) == 1:
                where_filter = {"doc_id": clean_ids[0]}
            elif len(clean_ids) > 1:
                where_filter = {"doc_id": {"$in": clean_ids}}

        # Generate query embedding
        query_vec = self.embedding_provider.embed_query(query_text)

        # Chroma requires n_results <= total collection count
        query_k = min(k, total_items)

        try:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=query_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query execution error: %s", exc)
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        passages: List[RetrievedPassage] = []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Distance in cosine space is between 0 (identical) and 2 (opposite)
            # Map distance to similarity score in [0.0, 1.0]
            similarity = round(max(0.0, min(1.0, 1.0 - (dist / 2.0))), 4)

            if similarity < threshold:
                continue

            passage = RetrievedPassage(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", "UNKNOWN"),
                filename=meta.get("filename", "UNKNOWN"),
                page_number=int(meta.get("page_number", 1)),
                text=text,
                score=similarity,
                metadata=meta,
            )
            passages.append(passage)

        # Sort descending by score
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages[:k]
