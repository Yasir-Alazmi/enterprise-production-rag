"""Unit tests for cross-encoder reranking."""

from src.ingestion.chunker import TextChunk
from src.reranker.cross_encoder import CrossEncoderReranker

def test_cross_encoder_rerank():
    reranker = CrossEncoderReranker(top_k=2)
    
    chunk_a = TextChunk(
        chunk_id="c1", document_id="doc1",
        content="The server SLA guarantees 99.95% availability for production systems.",
        chunk_index=0, token_count=10
    )
    chunk_b = TextChunk(
        chunk_id="c2", document_id="doc1",
        content="Office cafeteria hours are from 8am to 4pm daily.",
        chunk_index=1, token_count=10
    )
    
    candidates = [(chunk_b, 0.5), (chunk_a, 0.4)]
    query = "What is the server SLA availability guarantee?"
    
    reranked = reranker.rerank(query, candidates)
    assert len(reranked) == 2
    # chunk_a should be reranked to position 1 due to high term overlap and coverage
    assert reranked[0][0].chunk_id == "c1"
    assert reranked[0][1] > reranked[1][1]
