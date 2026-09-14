"""Bounded LRU semantic query cache with memory protection."""

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    query: str
    response: str
    citations: List[Dict[str, Any]]
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)

class SemanticCache:
    """Bounded vector cache with LRU eviction and cosine threshold matching."""

    def __init__(self, similarity_threshold: float = 0.92, ttl_seconds: int = 3600, max_entries: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache: OrderedDict[str, Tuple[np.ndarray, CacheEntry]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def lookup(self, query: str, query_embedding: np.ndarray) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """Look up similar past queries. Moves accessed key to most-recently-used on hit."""
        now = time.time()

        # Purge expired entries
        expired_keys = [k for k, (_, entry) in self.cache.items() if now - entry.created_at > self.ttl_seconds]
        for k in expired_keys:
            del self.cache[k]

        if not self.cache or query_embedding is None:
            self.misses += 1
            return None

        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-6:
            self.misses += 1
            return None

        best_score = -1.0
        best_key: Optional[str] = None
        best_entry: Optional[CacheEntry] = None

        for key, (cached_vec, entry) in self.cache.items():
            score = float(np.dot(query_embedding, cached_vec))
            if score > best_score:
                best_score = score
                best_key = key
                best_entry = entry

        if best_score >= self.similarity_threshold and best_entry is not None and best_key is not None:
            self.hits += 1
            best_entry.last_accessed = now
            self.cache.move_to_end(best_key)  # Mark as recently used
            return best_entry.response, best_entry.citations, round(best_score, 4)

        self.misses += 1
        return None

    def store(self, query: str, query_embedding: np.ndarray, response: str, citations: List[Dict[str, Any]]) -> None:
        """Store a new query-response pair, evicting the least recently used entry if full."""
        if query_embedding is None:
            return

        # Enforce capacity via LRU eviction
        if len(self.cache) >= self.max_entries:
            self.cache.popitem(last=False)  # Evict oldest
            self.evictions += 1

        entry = CacheEntry(
            query=query,
            response=response,
            citations=citations,
            created_at=time.time(),
            last_accessed=time.time()
        )
        self.cache[query] = (query_embedding, entry)

    def stats(self) -> Dict[str, Any]:
        """Return cache health and eviction statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "entries_count": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 4),
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }

    def clear(self) -> None:
        """Clear cache state."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
