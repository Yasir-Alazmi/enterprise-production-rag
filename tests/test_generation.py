"""Unit tests for generative multi-source synthesis and hallucination defense."""

from src.generation.generator import EnterpriseSynthesisGenerator
from src.ingestion.chunker import TextChunk


def test_generation_multi_source_synthesis():
    generator = EnterpriseSynthesisGenerator(min_relevance_threshold=0.20)

    chunk_a = TextChunk(
        chunk_id="sla_01", document_id="cloud_sla",
        content="Production cloud systems guarantee 99.95% monthly uptime. Automated failover triggers within 120 seconds.",
        chunk_index=0, token_count=15,
        metadata={"document_title": "Cloud SLA", "section": "Availability"}
    )
    chunk_b = TextChunk(
        chunk_id="sec_01", document_id="cloud_sec",
        content="All data at rest requires AES-256 encryption with customer keys.",
        chunk_index=0, token_count=12,
        metadata={"document_title": "Cloud Security", "section": "Encryption"}
    )

    candidates = [(chunk_a, 0.85), (chunk_b, 0.70)]
    query = "What is the monthly uptime and encryption requirement for production cloud?"

    result = generator.generate_answer(query, candidates)
    assert result.grounded is True
    assert "Based on verified enterprise policies" in result.answer
    assert "99.95%" in result.answer
    assert "AES-256" in result.answer
    assert "[Cloud SLA - Availability]" in result.answer
    assert len(result.cited_sections) >= 1

def test_generation_insufficient_context_refusal():
    generator = EnterpriseSynthesisGenerator(min_relevance_threshold=0.20)

    chunk = TextChunk(
        chunk_id="c1", document_id="doc1",
        content="The cafeteria serves sandwiches and hot coffee from 8am to 4pm.",
        chunk_index=0, token_count=12
    )

    # Completely unrelated query
    candidates = [(chunk, 0.10)]
    query = "What is the quantum computing qubit error threshold?"

    result = generator.generate_answer(query, candidates)
    assert result.grounded is False
    assert "Insufficient enterprise documentation was retrieved" in result.answer

def test_generation_empty_candidates_returns_refusal():
    generator = EnterpriseSynthesisGenerator()
    result = generator.generate_answer("Any query?", [])
    assert result.grounded is False
    assert "Insufficient enterprise documentation was retrieved" in result.answer

