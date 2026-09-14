"""Dual-mode embedding engine: Deterministic hashed projection & Neural embeddings."""

import hashlib
from typing import List, Optional, Protocol

import numpy as np

from src.core.logging import get_logger
from src.retrieval.sparse_search import BM25Index

logger = get_logger(__name__)


class BaseEmbeddingEngine(Protocol):
    """Protocol interface defining the embedding model contract."""

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text into normalized dense vector."""
        ...

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch of texts into 2D normalized array (N, dimension)."""
        ...


class DeterministicHashingEmbedding:
    """Zero-dependency, reproducible deterministic pseudo-semantic projection.

    Uses tokenized SHA256 hashed sign-projection into a D-dimensional hypersphere.
    Designed for ultra-fast local execution, offline CI matrices, and deterministic benchmarks.
    """

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        tokens = BM25Index.tokenize(text)
        vec = np.zeros(self._dimension, dtype=np.float32)
        if not tokens:
            return vec

        for idx, token in enumerate(tokens):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            pos = h % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            weight = 1.0 / (1.0 + 0.05 * idx)
            vec[pos] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.embed_text(t) for t in texts])


class SentenceTransformerEmbedding:
    """Neural dense vector embedding adapter supporting Sentence Transformers.

    Falls back cleanly to DenseSemanticProjection if external neural weights
    are unavailable in lightweight environments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self._model_name = model_name
        self._dimension = dimension
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded neural embedding model: %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; activating DenseSemanticProjection fallback."
            )
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        if self._model is not None:
            emb = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return emb.astype(np.float32)

        # Fallback projection for environments without sentence-transformers
        hashing = DeterministicHashingEmbedding(dimension=self._dimension)
        return hashing.embed_text(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if self._model is not None:
            embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embs.astype(np.float32)
        return np.vstack([self.embed_text(t) for t in texts])


def get_embedding_engine(
    provider: Optional[str] = None,
    dimension: Optional[int] = None,
) -> BaseEmbeddingEngine:
    """Factory function instantiating the configured embedding engine."""
    prov = (provider or "deterministic").lower().strip()
    dim = dimension or (384 if prov in ["neural", "sentence-transformers"] else 128)

    if prov in ["neural", "sentence-transformers"]:
        return SentenceTransformerEmbedding(dimension=dim)
    return DeterministicHashingEmbedding(dimension=dim)
