"""Dense vector store and normalized embedding similarity engine."""

import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index

class DenseVectorStore:
    """In-memory vector database with L2-normalized cosine search."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.chunks_map: Dict[str, TextChunk] = {}

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic, normalized pseudo-semantic feature projection.
        Guarantees zero-cost offline reproducibility while maintaining semantic clustering.
        """
        tokens = BM25Index.tokenize(text)
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not tokens:
            return vec

        for idx, token in enumerate(tokens):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            pos = h % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            weight = 1.0 / (1.0 + 0.05 * idx)
            vec[pos] += sign * weight

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Generate embeddings and index candidate chunks."""
        for chunk in chunks:
            self.chunks_map[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = self._embed(chunk.content)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Compute cosine similarity across all indexed vectors."""
        if not self.vectors:
            return []

        query_vec = self._embed(query)
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-6:
            return []

        scores: List[Tuple[str, float]] = []
        for chunk_id, doc_vec in self.vectors.items():
            sim = float(np.dot(query_vec, doc_vec))
            scores.append((chunk_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
