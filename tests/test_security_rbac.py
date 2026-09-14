"""Unit tests for Role-Based Access Control and multi-tenant document isolation."""

import pytest

from src.core.security import AccessControlManager, ClassificationLevel
from src.ingestion.chunker import TextChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore


@pytest.fixture
def classified_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id="pub_01", document_id="doc_pub",
            content="Welcome to our public company website and products overview.",
            chunk_index=0, token_count=10, classification="PUBLIC"
        ),
        TextChunk(
            chunk_id="int_01", document_id="doc_int",
            content="The internal corporate cafeteria hours are 8am to 4pm daily.",
            chunk_index=0, token_count=10, classification="INTERNAL"
        ),
        TextChunk(
            chunk_id="conf_01", document_id="doc_conf",
            content="Security encryption keys must rotate automatically every 90 days in the HSM.",
            chunk_index=0, token_count=12, classification="CONFIDENTIAL"
        ),
        TextChunk(
            chunk_id="rest_01", document_id="doc_rest",
            content="Executive payroll and executive bonus distribution records for Q4.",
            chunk_index=0, token_count=10, classification="RESTRICTED"
        )
    ]

def test_clearance_levels():
    assert AccessControlManager.can_access("guest", "PUBLIC") is True
    assert AccessControlManager.can_access("guest", "INTERNAL") is False
    assert AccessControlManager.can_access("guest", "RESTRICTED") is False

    assert AccessControlManager.can_access("employee", "PUBLIC") is True
    assert AccessControlManager.can_access("employee", "INTERNAL") is True
    assert AccessControlManager.can_access("employee", "CONFIDENTIAL") is False

    assert AccessControlManager.can_access("admin", "RESTRICTED") is True

def test_rbac_retrieval_isolation(classified_chunks: list[TextChunk]):
    bm25 = BM25Index()
    bm25.index_chunks(classified_chunks)
    vec = DenseVectorStore(dimension=128)
    vec.index_chunks(classified_chunks)
    retriever = HybridRetriever(bm25, vec)

    # Guest user searching for security keys
    guest_results = retriever.search("security encryption keys HSM", user_role="guest")
    # Guest should get ZERO confidential chunks
    assert not any(c.classification in ["CONFIDENTIAL", "RESTRICTED"] for c, _ in guest_results)

    # Admin user searching for security keys
    admin_results = retriever.search("security encryption keys HSM", user_role="admin")
    assert any(c.chunk_id == "conf_01" for c, _ in admin_results)

def test_unknown_role_defaults_to_internal():
    # Unknown roles default safely to INTERNAL clearance
    assert AccessControlManager.get_user_clearance("unknown_external_agent") == ClassificationLevel.INTERNAL
    assert AccessControlManager.can_access("unknown_external_agent", "RESTRICTED") is False
    assert AccessControlManager.can_access("unknown_external_agent", "INTERNAL") is True

def test_hierarchical_clearance_inheritance():
    # RESTRICTED clearance inherits access to ALL lower levels
    assert AccessControlManager.can_access("executive", "PUBLIC") is True
    assert AccessControlManager.can_access("executive", "INTERNAL") is True
    assert AccessControlManager.can_access("executive", "CONFIDENTIAL") is True
    assert AccessControlManager.can_access("executive", "RESTRICTED") is True


def test_fail_fast_production_security_validation():
    from src.core.config import Settings

    # Insecure default secret in production environment must fail fast
    prod_insecure = Settings(app_env="production")
    with pytest.raises(ValueError) as excinfo:
        prod_insecure.validate_production_security()
    assert "FATAL SECURITY VIOLATION" in str(excinfo.value)


def test_valid_production_security_validation():
    from src.core.config import Settings

    # Secure custom key in production environment succeeds
    prod_secure = Settings(
        app_env="production",
        jwt_secret_key="a-truly-cryptographic-and-secure-high-entropy-jwt-secret-key-12345"
    )
    # Should not raise
    prod_secure.validate_production_security()

