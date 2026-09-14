"""Hybrid retriever fusing Dense and Sparse rankings with RBAC filtering."""

from typing import Dict, List, Optional, Tuple

from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore


class HybridRetriever:
    """Combines BM25 lexical precision with dense semantic recall under role-based access control."""

    def __init__(self, bm25_index: BM25Index, vector_store: DenseVectorStore, alpha: float = 0.60):
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.alpha = alpha

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_role: Optional[str] = None
    ) -> List[Tuple[TextChunk, float]]:
        """Execute hybrid retrieval with Reciprocal Rank Fusion and RBAC filtering."""
        sparse_results = self.bm25_index.search(query, top_k=top_k * 2, user_role=user_role)
        dense_results = self.vector_store.search(query, top_k=top_k * 2, user_role=user_role)

        rrf_scores: Dict[str, float] = {}
        rrf_k = 60.0

        for rank, (chunk_id, _) in enumerate(sparse_results):
            score = (1.0 - self.alpha) / (rrf_k + rank + 1.0)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        for rank, (chunk_id, _) in enumerate(dense_results):
            score = self.alpha / (rrf_k + rank + 1.0)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        output: List[Tuple[TextChunk, float]] = []
        for chunk_id, fused_score in ranked[:top_k]:
            chunk = self.vector_store.chunks_map.get(chunk_id) or self.bm25_index.chunks_map.get(chunk_id)
            if chunk:
                output.append((chunk, round(fused_score, 6)))

        return output
