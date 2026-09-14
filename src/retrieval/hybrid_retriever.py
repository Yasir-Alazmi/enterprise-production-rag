"""Hybrid retriever fusing Dense and Sparse rankings via Reciprocal Rank Fusion (RRF)."""

from typing import Dict, List, Tuple
from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore

class HybridRetriever:
    """Combines BM25 lexical precision with dense semantic recall."""

    def __init__(self, bm25_index: BM25Index, vector_store: DenseVectorStore, alpha: float = 0.60):
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.alpha = alpha  # alpha weight for dense, (1 - alpha) for sparse

    def search(self, query: str, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """Execute hybrid retrieval with Reciprocal Rank Fusion."""
        sparse_results = self.bm25_index.search(query, top_k=top_k * 2)
        dense_results = self.vector_store.search(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        rrf_k = 60.0  # standard RRF constant

        # Accumulate sparse rank scores
        for rank, (chunk_id, _) in enumerate(sparse_results):
            score = (1.0 - self.alpha) / (rrf_k + rank + 1.0)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        # Accumulate dense rank scores
        for rank, (chunk_id, _) in enumerate(dense_results):
            score = self.alpha / (rrf_k + rank + 1.0)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        # Sort candidate chunks by fused score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        output: List[Tuple[TextChunk, float]] = []
        for chunk_id, fused_score in ranked[:top_k]:
            chunk = self.vector_store.chunks_map.get(chunk_id) or self.bm25_index.chunks_map.get(chunk_id)
            if chunk:
                output.append((chunk, round(fused_score, 6)))

        return output
