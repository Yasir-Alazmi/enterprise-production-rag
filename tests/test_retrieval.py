"""Unit tests for BM25, Dense Vector Store, and Hybrid RRF Retriever."""

from src.ingestion.chunker import TextChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore


def test_bm25_lexical_search(sample_chunks: list[TextChunk]):
    index = BM25Index()
    index.index_chunks(sample_chunks)

    results = index.search("AES-256 encryption", top_k=2)
    assert len(results) > 0
    top_chunk_id, score = results[0]
    assert score > 0.0
    matched_chunk = index.chunks_map[top_chunk_id]
    assert "AES-256" in matched_chunk.content or "encryption" in matched_chunk.content.lower()

def test_bm25_no_matches(sample_chunks: list[TextChunk]):
    index = BM25Index()
    index.index_chunks(sample_chunks)
    results = index.search("nonexistent_random_term_xyz_123")
    assert len(results) == 0

def test_dense_vector_store_cosine_similarity(sample_chunks: list[TextChunk]):
    store = DenseVectorStore(dimension=128)
    store.index_chunks(sample_chunks)

    results = store.search("uptime availability microservices", top_k=2)
    assert len(results) > 0
    top_id, sim = results[0]
    assert -1.0 <= sim <= 1.0

def test_hybrid_rrf_retrieval(populated_retriever: HybridRetriever):
    results = populated_retriever.search("incident response P1 15 minutes", top_k=2)
    assert len(results) > 0
    chunk, rrf_score = results[0]
    assert rrf_score > 0.0
    assert chunk.document_id == "cloud_policy_01"
