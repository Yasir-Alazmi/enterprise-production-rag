"""Unit tests for semantic query cache including LRU eviction."""

import numpy as np

from src.cache.semantic_cache import SemanticCache


def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=10)

    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2_close = np.array([0.98, 0.05, 0.0], dtype=np.float32)
    vec2_close /= np.linalg.norm(vec2_close)
    vec3_different = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    miss = cache.lookup("What is the SLA?", vec1)
    assert miss is None
    assert cache.misses == 1

    cache.store("What is the SLA?", vec1, "99.95% uptime", [{"doc": "sla"}])

    hit = cache.lookup("Can you tell me the SLA?", vec2_close)
    assert hit is not None
    response, citations, score = hit
    assert response == "99.95% uptime"
    assert score >= 0.90
    assert cache.hits == 1

    miss2 = cache.lookup("How to cook pasta?", vec3_different)
    assert miss2 is None
    assert cache.misses == 2

def test_lru_cache_eviction():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60, max_entries=2)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    cache.store("q1", vec, "a1", [])
    cache.store("q2", vec, "a2", [])
    assert len(cache.cache) == 2
    assert cache.evictions == 0

    # Adding third entry should trigger LRU eviction of q1
    cache.store("q3", vec, "a3", [])
    assert len(cache.cache) == 2
    assert cache.evictions == 1
    assert "q1" not in cache.cache
    assert "q2" in cache.cache
    assert "q3" in cache.cache
