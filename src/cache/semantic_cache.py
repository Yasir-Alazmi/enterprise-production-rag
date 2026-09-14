"""Semantic similarity query cache to optimize latency and operational cost."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    query: str
    response: str
    citations: List[Dict[str, Any]]
    created_at: float = Field(default_factory=time.time)

class SemanticCache:
    """Vector-based cache matching semantically equivalent queries."""

    def __init__(self, similarity_threshold: float = 0.92, ttl_seconds: int = 3600):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.cache: List[Tuple[np.ndarray, CacheEntry]] = []
        self.hits: int = 0
        self.misses: int = 0

    def lookup(self, query: str, query_embedding: np.ndarray) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """Look up similar past queries. Returns (response, citations, similarity) if hit."""
        now = time.time()

        # Purge expired entries
        self.cache = [(vec, ent) for vec, ent in self.cache if now - ent.created_at <= self.ttl_seconds]

        if not self.cache or query_embedding is None:
            self.misses += 1
            return None

        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-6:
            self.misses += 1
            return None

        best_score = -1.0
        best_entry: Optional[CacheEntry] = None

        for cached_vec, entry in self.cache:
            score = float(np.dot(query_embedding, cached_vec))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.similarity_threshold and best_entry is not None:
            self.hits += 1
            return best_entry.response, best_entry.citations, round(best_score, 4)

        self.misses += 1
        return None

    def store(self, query: str, query_embedding: np.ndarray, response: str, citations: List[Dict[str, Any]]) -> None:
        """Store a new query-response pair in the semantic cache."""
        if query_embedding is None:
            return

        entry = CacheEntry(
            query=query,
            response=response,
            citations=citations,
            created_at=time.time()
        )
        self.cache.append((query_embedding, entry))

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "entries_count": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }

    def clear(self) -> None:
        """Empty the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
