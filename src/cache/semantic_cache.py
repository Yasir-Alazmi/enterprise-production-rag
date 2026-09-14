"""Bounded LRU semantic query cache with Role-Based Access Control scoping."""

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    query: str
    response: str
    citations: List[Dict[str, Any]]
    clearance_level: int = Field(default=1, description="ClassificationLevel integer (0=PUBLIC, 1=INTERNAL, 2=CONFIDENTIAL, 3=RESTRICTED)")
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)

class SemanticCache:
    """Role-scoped bounded vector cache with LRU eviction and zero cross-clearance leakage."""

    def __init__(self, similarity_threshold: float = 0.92, ttl_seconds: int = 3600, max_entries: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache: OrderedDict[str, Tuple[np.ndarray, CacheEntry]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.clearance_denials: int = 0

    def lookup(
        self,
        query: str,
        query_embedding: np.ndarray,
        user_clearance: int = 1
    ) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """Look up similar past queries, strictly enforcing that user_clearance >= cached clearance_level."""
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
            # Enforce Role Clearance Scoping
            if user_clearance < best_entry.clearance_level:
                # User has insufficient clearance to view this cached answer!
                self.clearance_denials += 1
                self.misses += 1
                return None

            self.hits += 1
            best_entry.last_accessed = now
            self.cache.move_to_end(best_key)
            return best_entry.response, best_entry.citations, round(best_score, 4)

        self.misses += 1
        return None

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        response: str,
        citations: List[Dict[str, Any]],
        clearance_level: int = 1
    ) -> None:
        """Store a query-response pair bound to its security clearance level."""
        if query_embedding is None:
            return

        if len(self.cache) >= self.max_entries:
            self.cache.popitem(last=False)
            self.evictions += 1

        entry = CacheEntry(
            query=query,
            response=response,
            citations=citations,
            clearance_level=clearance_level,
            created_at=time.time(),
            last_accessed=time.time()
        )
        self.cache[query] = (query_embedding, entry)

    def stats(self) -> Dict[str, Any]:
        """Return cache health, evictions, and clearance denial statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "entries_count": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "clearance_denials": self.clearance_denials,
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
        self.clearance_denials = 0
