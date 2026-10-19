"""
Embedding generation providers for Sentence Transformers and deterministic test mocks.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Protocol
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("embeddings")


class EmbeddingProvider(Protocol):
    """Protocol defining the embedding provider interface."""

    dimension: int

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text strings."""
        ...

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding for a single search query string."""
        ...


class SentenceTransformerEmbeddings:
    """
    Embedding provider powered by Sentence-Transformers.
    Employs lazy-loading so importing does not immediately trigger model download or memory allocation.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding_model_name
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazy loads the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Force lazy load to determine dimension
            _ = self.model
        return self._dimension or 384

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        if not query.strip():
            return [0.0] * self.dimension
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


class DeterministicMockEmbeddings:
    """
    Fast, deterministic embedding generator for unit tests.
    Produces unit-normalized vectors without downloading external neural weights.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        """Generates a deterministic float vector from text hash and normalizes it."""
        if not text.strip():
            return [0.0] * self.dimension

        # Generate a seed value from SHA-256
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec: List[float] = []

        # Derive dimension coordinates deterministically
        for i in range(self.dimension):
            byte_val = h[(i * 7) % len(h)]
            # Map byte (0-255) to float [-1.0, 1.0] with sine harmonic
            coord = ((byte_val / 127.5) - 1.0) * math.cos(i * 0.1)
            vec.append(coord)

        # L2-normalize vector to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._hash_to_vector(query)


def get_embedding_provider(
    model_name: Optional[str] = None,
    use_mock: bool = False,
) -> EmbeddingProvider:
    """Factory creating the appropriate embedding provider."""
    if use_mock:
        return DeterministicMockEmbeddings()
    return SentenceTransformerEmbeddings(model_name=model_name)
