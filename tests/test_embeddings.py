"""Unit tests for pluggable embedding engines."""

import numpy as np
import pytest

from src.retrieval.embeddings import (
    DeterministicHashingEmbedding,
    SentenceTransformerEmbedding,
    get_embedding_engine,
)
from src.retrieval.vector_store import DenseVectorStore


def test_deterministic_hashing_embedding_dimension():
    engine = DeterministicHashingEmbedding(dimension=128)
    assert engine.dimension == 128
    vec = engine.embed_text("Cloud security infrastructure")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (128,)
    # Verify unit-norm normalization
    assert pytest.approx(np.linalg.norm(vec), abs=1e-5) == 1.0


def test_deterministic_hashing_consistency():
    engine = DeterministicHashingEmbedding(dimension=64)
    v1 = engine.embed_text("Incident response SLA 15 minutes")
    v2 = engine.embed_text("Incident response SLA 15 minutes")
    assert np.allclose(v1, v2)

    # Empty string returns zero vector
    v_empty = engine.embed_text("")
    assert np.allclose(v_empty, np.zeros(64, dtype=np.float32))


def test_batch_embedding():
    engine = DeterministicHashingEmbedding(dimension=32)
    texts = ["Document one", "Document two", "Document three"]
    matrix = engine.embed_batch(texts)
    assert matrix.shape == (3, 32)


def test_sentence_transformer_fallback():
    # In environments without sentence_transformers, falls back gracefully
    engine = SentenceTransformerEmbedding(dimension=256)
    assert engine.dimension == 256
    vec = engine.embed_text("Enterprise access control")
    assert vec.shape == (256,)


def test_embedding_factory():
    engine_det = get_embedding_engine("deterministic", dimension=64)
    assert engine_det.dimension == 64

    engine_neural = get_embedding_engine("neural", dimension=384)
    assert engine_neural.dimension == 384


def test_vector_store_with_custom_engine():
    custom_engine = DeterministicHashingEmbedding(dimension=64)
    store = DenseVectorStore(dimension=64, embedding_engine=custom_engine)
    assert store.dimension == 64
    vec = store._embed("Sample text")
    assert vec.shape == (64,)
