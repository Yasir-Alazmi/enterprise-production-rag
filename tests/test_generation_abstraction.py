"""Unit tests for Answer Generator Abstraction and Pluggable Providers."""

from src.generation.generator import (
    DeterministicGroundedGenerator,
    EnterpriseSynthesisGenerator,
    LLMAnswerGenerator,
    get_answer_generator,
)
from src.ingestion.chunker import TextChunk


def test_deterministic_grounded_generator_insufficient_context():
    gen = DeterministicGroundedGenerator(min_relevance_threshold=0.5)
    res = gen.generate_answer(query="What is quantum teleportation?", retrieved_chunks=[])
    assert res.grounded is False
    assert "Insufficient enterprise documentation" in res.answer


def test_llm_generator_fallback_without_api_key():
    llm_gen = LLMAnswerGenerator(provider="openai", api_key=None)
    chunk = TextChunk(
        chunk_id="c1",
        document_id="doc1",
        content="System password policy requires minimum 14 characters.",
        chunk_index=0,
        token_count=10,
        classification="INTERNAL",
        metadata={"document_title": "Security Policy", "section": "Passwords"}
    )
    # Without api_key, must seamlessly fall back to deterministic grounded synthesis
    res = llm_gen.generate_answer("What is the password requirement?", [(chunk, 0.85)])
    assert res.grounded is True
    assert "14 characters" in res.answer
    assert "[Security Policy - Passwords]" in res.answer


def test_generator_factory():
    gen_det = get_answer_generator("deterministic")
    assert isinstance(gen_det, DeterministicGroundedGenerator)

    gen_llm = get_answer_generator("openai")
    assert isinstance(gen_llm, LLMAnswerGenerator)

    # Verify backward compatibility alias
    assert EnterpriseSynthesisGenerator is DeterministicGroundedGenerator


def test_ollama_generator_fallback_on_unreachable_endpoint():
    # Calling Ollama with unreachable port falls back cleanly to deterministic synthesis
    ollama_gen = LLMAnswerGenerator(provider="ollama", base_url="http://127.0.0.1:59999")
    chunk = TextChunk(
        chunk_id="c_ollama",
        document_id="doc_ollama",
        content="Incident response SLA for P1 outages requires 15 minutes.",
        chunk_index=0,
        token_count=10,
        classification="INTERNAL",
        metadata={"document_title": "Incident Protocol", "section": "SLAs"}
    )
    res = ollama_gen.generate_answer("What is the P1 incident response SLA?", [(chunk, 0.90)])
    assert res.grounded is True
    assert "15 minutes" in res.answer
    assert "[Incident Protocol - SLAs]" in res.answer

