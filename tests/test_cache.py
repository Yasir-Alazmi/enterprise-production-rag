"""Unit tests for semantic query cache."""

import numpy as np

from src.cache.semantic_cache import SemanticCache


def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=60)

    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2_close = np.array([0.98, 0.05, 0.0], dtype=np.float32)
    vec2_close /= np.linalg.norm(vec2_close)
    vec3_different = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 1. Initial lookup -> Miss
    miss = cache.lookup("What is the SLA?", vec1)
    assert miss is None
    assert cache.misses == 1

    # 2. Store answer
    cache.store("What is the SLA?", vec1, "99.95% uptime", [{"doc": "sla"}])

    # 3. Lookup with semantically close vector -> Hit
    hit = cache.lookup("Can you tell me the SLA?", vec2_close)
    assert hit is not None
    response, citations, score = hit
    assert response == "99.95% uptime"
    assert score >= 0.90
    assert cache.hits == 1

    # 4. Lookup with completely different vector -> Miss
    miss2 = cache.lookup("How to cook pasta?", vec3_different)
    assert miss2 is None
    assert cache.misses == 2
