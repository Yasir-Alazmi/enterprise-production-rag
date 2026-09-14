"""Shared pytest fixtures."""

import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.ingestion.chunker import RecursiveTokenChunker, TextChunk
from src.ingestion.parser import Document
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id="cloud_policy_01",
        title="Enterprise Cloud Security",
        content="""# Enterprise Cloud Infrastructure
Availability requires 99.95% monthly uptime.
Encryption at rest must use AES-256 with customer-managed keys.
Data in transit requires TLS 1.3 across all communication channels.
Incident response for P1 critical events must occur within 15 minutes.
Disaster recovery requires an RPO of 1 hour and RTO of 4 hours.""",
        metadata={"category": "security"}
    )

@pytest.fixture
def sample_chunks(sample_document: Document) -> list[TextChunk]:
    chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=20)
    return chunker.split_document(sample_document)

@pytest.fixture
def populated_retriever(sample_chunks: list[TextChunk]) -> HybridRetriever:
    bm25 = BM25Index()
    bm25.index_chunks(sample_chunks)
    vec = DenseVectorStore(dimension=128)
    vec.index_chunks(sample_chunks)
    return HybridRetriever(bm25_index=bm25, vector_store=vec, alpha=0.60)

@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)
