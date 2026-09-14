"""Generative Answer Synthesis abstraction supporting deterministic grounding and LLM providers."""

import re
from typing import List, Optional, Protocol, Tuple

import httpx
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.logging import get_logger
from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index

logger = get_logger(__name__)


class GenerationResult(BaseModel):
    answer: str
    grounded: bool
    relevance_score: float
    cited_sections: List[str] = Field(default_factory=list)
    provider: str = Field(default="deterministic-grounded")


class BaseAnswerGenerator(Protocol):
    """Protocol interface defining the RAG answer synthesis contract."""

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[TextChunk, float]]
    ) -> GenerationResult:
        """Synthesize response from retrieved chunks with citations and strict grounding."""
        ...


class DeterministicGroundedGenerator:
    """Deterministic, citation-grounded extractive synthesis engine.

    Synthesizes factual natural language responses from multiple retrieved chunks without
    external API calls, hallucinations, or non-deterministic variance. Ideal for offline CI,
    compliance testing, and ultra-low latency benchmarks.
    """

    INSUFFICIENT_CONTEXT_REFUSAL = (
        "Insufficient enterprise documentation was retrieved to answer this query with verifiable certainty."
    )

    def __init__(self, min_relevance_threshold: float = 0.20):
        self.min_relevance_threshold = min_relevance_threshold

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[TextChunk, float]]
    ) -> GenerationResult:
        """Synthesize a cohesive, citation-grounded response across retrieved chunks."""
        if not retrieved_chunks:
            return GenerationResult(
                answer=self.INSUFFICIENT_CONTEXT_REFUSAL,
                grounded=False,
                relevance_score=0.0,
                provider="deterministic-grounded"
            )

        top_chunk, top_score = retrieved_chunks[0]
        if top_score < self.min_relevance_threshold:
            logger.info("Top retrieval score %.4f below minimum threshold %.4f. Refusing.", top_score, self.min_relevance_threshold)
            return GenerationResult(
                answer=self.INSUFFICIENT_CONTEXT_REFUSAL,
                grounded=False,
                relevance_score=top_score,
                provider="deterministic-grounded"
            )

        STOP_WORDS = {
            "a", "an", "the", "and", "or", "but", "if", "as", "what",
            "which", "this", "that", "these", "those", "for", "is", "of",
            "to", "from", "in", "out", "on", "by", "with", "are", "was",
            "were", "be", "been", "have", "has", "had", "do", "does", "did"
        }
        query_tokens = {tok for tok in BM25Index.tokenize(query) if tok not in STOP_WORDS}
        if not query_tokens:
            query_tokens = set(BM25Index.tokenize(query))

        synthesized_points: List[str] = []
        cited_sections: List[str] = []

        for chunk, score in retrieved_chunks:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.content) if len(s.strip()) > 15]

            relevant_sentences = []
            for sent in sentences:
                sent_tokens = {tok for tok in BM25Index.tokenize(sent) if tok not in STOP_WORDS}
                intersection = query_tokens.intersection(sent_tokens)
                if len(intersection) >= 1:
                    relevant_sentences.append((sent, len(intersection)))

            relevant_sentences.sort(key=lambda x: x[1], reverse=True)

            if relevant_sentences:
                best_sent, match_count = relevant_sentences[0]
                if match_count < 1:
                    continue

                doc_title = chunk.metadata.get("document_title", chunk.document_id)
                sec = chunk.metadata.get("section", "General")
                source_tag = f"[{doc_title} - {sec}]"

                if source_tag not in cited_sections:
                    cited_sections.append(source_tag)

                # Avoid duplicate factual statements across chunks
                if not any(best_sent.lower() == p.split(" [")[0].lower() for p in synthesized_points):
                    synthesized_points.append(f"{best_sent} {source_tag}")

        if not synthesized_points:
            return GenerationResult(
                answer=self.INSUFFICIENT_CONTEXT_REFUSAL,
                grounded=False,
                relevance_score=top_score,
                provider="deterministic-grounded"
            )

        lead_summary = f"Based on verified enterprise policies, {synthesized_points[0]}"
        if len(synthesized_points) > 1:
            additional_context = " Furthermore, " + " Additionally, ".join(synthesized_points[1:])
            full_answer = lead_summary + additional_context
        else:
            full_answer = lead_summary

        return GenerationResult(
            answer=full_answer,
            grounded=True,
            relevance_score=top_score,
            cited_sections=cited_sections,
            provider="deterministic-grounded"
        )


# Backward compatibility alias
EnterpriseSynthesisGenerator = DeterministicGroundedGenerator


class LLMAnswerGenerator:
    """Pluggable live LLM generator supporting OpenAI, Azure, and local Ollama instances."""

    SYSTEM_PROMPT = (
        "You are an Enterprise RAG Assistant. Answer the query strictly and exclusively using "
        "the provided Context chunks. Every factual statement must cite its source section in brackets, "
        "e.g., [Document Title - Section]. If the context is insufficient or missing, output exactly: "
        "'" + DeterministicGroundedGenerator.INSUFFICIENT_CONTEXT_REFUSAL + "'"
    )

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback_generator: Optional[DeterministicGroundedGenerator] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.base_url = base_url or settings.ollama_base_url
        self.fallback = fallback_generator or DeterministicGroundedGenerator()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[TextChunk, float]]
    ) -> GenerationResult:
        """Call LLM provider with grounding context, falling back gracefully on network or auth errors."""
        if not retrieved_chunks:
            return self.fallback.generate_answer(query, retrieved_chunks)

        # Build grounded context block
        context_blocks = []
        cited_sections = []
        for idx, (chunk, score) in enumerate(retrieved_chunks, start=1):
            doc_title = chunk.metadata.get("document_title", chunk.document_id)
            sec = chunk.metadata.get("section", "General")
            source_tag = f"[{doc_title} - {sec}]"
            if source_tag not in cited_sections:
                cited_sections.append(source_tag)
            context_blocks.append(f"Context [{idx}] Source: {source_tag}\n{chunk.content}")

        context_str = "\n\n".join(context_blocks)
        user_prompt = f"Query: {query}\n\nContext:\n{context_str}\n\nAnswer:"

        if self.provider == "openai" and self.api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        llm_answer = data["choices"][0]["message"]["content"].strip()
                        return GenerationResult(
                            answer=llm_answer,
                            grounded=True,
                            relevance_score=retrieved_chunks[0][1],
                            cited_sections=cited_sections,
                            provider=f"openai:{self.model}"
                        )
            except Exception as e:
                logger.warning("Live OpenAI generation call failed (%s); falling back to deterministic synthesis.", e)

        elif self.provider == "ollama":
            try:
                payload = {
                    "model": self.model if self.model != "gpt-4o-mini" else "llama3.2",
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                url = f"{self.base_url.rstrip('/')}/api/chat"
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        llm_answer = data.get("message", {}).get("content", "").strip()
                        if llm_answer:
                            return GenerationResult(
                                answer=llm_answer,
                                grounded=True,
                                relevance_score=retrieved_chunks[0][1],
                                cited_sections=cited_sections,
                                provider=f"ollama:{payload['model']}"
                            )
            except Exception as e:
                logger.warning("Live Ollama generation call failed (%s); falling back to deterministic synthesis.", e)

        # Default fallback to deterministic grounding engine
        res = self.fallback.generate_answer(query, retrieved_chunks)
        return res


def get_answer_generator(provider: Optional[str] = None) -> BaseAnswerGenerator:
    """Factory creating the configured answer generator."""
    prov = (provider or settings.generation_provider).lower().strip()
    if prov in ["openai", "ollama", "llm"]:
        return LLMAnswerGenerator(provider=prov)
    return DeterministicGroundedGenerator()
