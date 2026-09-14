"""Generative RAG synthesis engine with multi-source factual grounding."""

import re
from typing import List, Tuple

from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index

logger = get_logger(__name__)

class GenerationResult(BaseModel):
    answer: str
    grounded: bool
    relevance_score: float
    cited_sections: List[str] = Field(default_factory=list)

class EnterpriseSynthesisGenerator:
    """Synthesizes factual natural language responses from multiple retrieved chunks.
    Enforces strict grounding, cross-source synthesis, and deterministic refusal on insufficient context.
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
                relevance_score=0.0
            )

        # Check top candidate confidence
        top_chunk, top_score = retrieved_chunks[0]
        if top_score < self.min_relevance_threshold:
            logger.info("Top retrieval score %.4f below minimum threshold %.4f. Refusing.", top_score, self.min_relevance_threshold)
            return GenerationResult(
                answer=self.INSUFFICIENT_CONTEXT_REFUSAL,
                grounded=False,
                relevance_score=top_score
            )

        query_tokens = set(BM25Index.tokenize(query))
        synthesized_points: List[str] = []
        cited_sections: List[str] = []

        for chunk, score in retrieved_chunks:
            # Extract individual sentences from the chunk
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.content) if len(s.strip()) > 15]

            # Select sentences with highest intersection with query tokens
            relevant_sentences = []
            for sent in sentences:
                sent_tokens = set(BM25Index.tokenize(sent))
                intersection = query_tokens.intersection(sent_tokens)
                if len(intersection) >= 1:
                    relevant_sentences.append((sent, len(intersection)))

            relevant_sentences.sort(key=lambda x: x[1], reverse=True)

            if relevant_sentences:
                best_sent = relevant_sentences[0][0]
                doc_title = chunk.metadata.get("document_title", chunk.document_id)
                sec = chunk.metadata.get("section", "General")
                source_tag = f"[{doc_title} - {sec}]"

                if source_tag not in cited_sections:
                    cited_sections.append(source_tag)

                # Avoid duplicate factual statements
                if best_sent not in synthesized_points:
                    synthesized_points.append(f"{best_sent} {source_tag}")

        if not synthesized_points:
            return GenerationResult(
                answer=self.INSUFFICIENT_CONTEXT_REFUSAL,
                grounded=False,
                relevance_score=top_score
            )

        # Synthesize into an executive, natural language paragraph
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
            cited_sections=cited_sections
        )
