"""Cross-encoder reranker scoring query-document pairs jointly."""

from typing import List, Tuple

from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index


class CrossEncoderReranker:
    """Eliminates dense false positives by evaluating joint text interactions."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def rerank(self, query: str, candidates: List[Tuple[TextChunk, float]]) -> List[Tuple[TextChunk, float]]:
        """Compute fine-grained interaction scores and sort candidates."""
        if not candidates:
            return []

        query_tokens = set(BM25Index.tokenize(query))
        reranked: List[Tuple[TextChunk, float]] = []

        for chunk, base_score in candidates:
            chunk_tokens = BM25Index.tokenize(chunk.content)
            if not chunk_tokens:
                reranked.append((chunk, 0.0))
                continue

            # 1. Exact query coverage ratio
            matched = query_tokens.intersection(set(chunk_tokens))
            coverage = len(matched) / max(1, len(query_tokens))

            # 2. Term density in chunk
            term_freq = sum(chunk_tokens.count(term) for term in query_tokens)
            density = term_freq / len(chunk_tokens)

            # 3. Position discount: earlier occurrences score higher
            first_pos = min([chunk_tokens.index(t) for t in matched] + [len(chunk_tokens)])
            pos_weight = 1.0 / (1.0 + 0.05 * first_pos)

            # Joint cross-interaction score
            raw_score = 0.5 * coverage + 0.3 * min(1.0, density * 5.0) + 0.2 * pos_weight

            # Blend with initial hybrid retrieval signal
            final_score = 0.7 * raw_score + 0.3 * min(1.0, base_score * 50.0)
            reranked.append((chunk, round(final_score, 4)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:self.top_k]
