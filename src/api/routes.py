"""API Route definitions and RAG pipeline execution with RBAC and Observability."""

import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Response, status

from src.api.schemas import (
    Citation,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.cache.semantic_cache import SemanticCache
from src.core.config import settings
from src.core.logging import get_logger
from src.core.metrics import metrics_collector
from src.evaluation.metrics import RAGEvaluator
from src.guardrails.injection_detector import InjectionDetector
from src.guardrails.pii_sanitizer import PIISanitizer
from src.ingestion.chunker import RecursiveTokenChunker, TextChunk
from src.ingestion.parser import Document
from src.reranker.cross_encoder import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "storage"

# Global pipeline instances
chunker = RecursiveTokenChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
bm25_index = BM25Index(k1=settings.sparse_k1, b=settings.sparse_b)
vector_store = DenseVectorStore(dimension=128)
hybrid_retriever = HybridRetriever(bm25_index=bm25_index, vector_store=vector_store, alpha=settings.hybrid_alpha)
reranker = CrossEncoderReranker(top_k=settings.rerank_top_k)
sanitizer = PIISanitizer()
injection_detector = InjectionDetector()
semantic_cache = SemanticCache(
    similarity_threshold=settings.cache_similarity_threshold,
    ttl_seconds=settings.cache_ttl_seconds,
    max_entries=1000
)
evaluator = RAGEvaluator()

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Verify service status, active indexes, and cache health."""
    return HealthResponse(
        indexed_chunks=len(vector_store.chunks_map),
        cache_entries=len(semantic_cache.cache),
        cache_stats=semantic_cache.stats()
    )

@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus operational telemetry."""
    text_content = metrics_collector.export_text(
        cache_stats=semantic_cache.stats(),
        indexed_chunks=len(vector_store.chunks_map)
    )
    return Response(content=text_content, media_type="text/plain; version=0.0.4; charset=utf-8")

@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_documents(payload: IngestRequest) -> IngestResponse:
    """Ingest, chunk, index, and optionally persist documents with RBAC classifications."""
    all_chunks: List[TextChunk] = []

    for doc_payload in payload.documents:
        doc = Document(
            id=doc_payload.id,
            title=doc_payload.title,
            content=doc_payload.content,
            classification=doc_payload.classification or "INTERNAL",
            metadata=doc_payload.metadata
        )
        chunks = chunker.split_document(doc)
        all_chunks.extend(chunks)

    if all_chunks:
        bm25_index.index_chunks(all_chunks)
        vector_store.index_chunks(all_chunks)

        if payload.persist_to_disk:
            vector_store.save_to_disk(STORAGE_DIR)
            bm25_index.save_to_disk(STORAGE_DIR)

    logger.info("Ingested %d documents (%d chunks)", len(payload.documents), len(all_chunks))
    return IngestResponse(
        indexed_documents=len(payload.documents),
        indexed_chunks=len(all_chunks),
        persisted=payload.persist_to_disk
    )

@router.post("/query", response_model=QueryResponse)
def query_pipeline(request: QueryRequest) -> QueryResponse:
    """Execute end-to-end RAG query with RBAC clearance, guardrails, and caching."""
    start_time = time.perf_counter()

    # 1. Adversarial Injection Screening
    if settings.enable_injection_detection:
        is_safe, violation = injection_detector.validate_prompt(request.query)
        if not is_safe:
            logger.warning("Rejected malicious prompt: %s", violation)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "SecurityViolation", "message": violation}
            )

    # 2. PII Sanitization
    sanitized_query = request.query
    detected_pii: List[str] = []
    was_sanitized = False
    if settings.enable_pii_masking:
        sanitized_query, was_sanitized, detected_pii = sanitizer.sanitize(request.query)

    # 3. Semantic Cache Check (scoped to identical queries)
    query_vec = vector_store._embed(sanitized_query)
    if request.enable_cache:
        cached_result = semantic_cache.lookup(sanitized_query, query_vec)
        if cached_result is not None:
            cached_ans, cached_cits_raw, _ = cached_result
            citations = [Citation(**c) for c in cached_cits_raw]
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return QueryResponse(
                query=request.query,
                answer=cached_ans,
                citations=citations,
                cached=True,
                latency_ms=round(elapsed_ms, 2),
                sanitized=was_sanitized,
                detected_pii=detected_pii
            )

    # 4. Hybrid Retrieval with RBAC filtering
    candidates = hybrid_retriever.search(
        sanitized_query,
        top_k=request.top_k * 2,
        user_role=request.user_role
    )

    if not candidates:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return QueryResponse(
            query=request.query,
            answer="No relevant documentation matching your security clearance was found.",
            citations=[],
            cached=False,
            latency_ms=round(elapsed_ms, 2),
            sanitized=was_sanitized,
            detected_pii=detected_pii
        )

    # 5. Cross-Encoder Reranking
    reranked = reranker.rerank(sanitized_query, candidates)

    # 6. Structured Grounded Answer Synthesis
    top_chunks = [chunk for chunk, _ in reranked[:request.top_k]]
    citations: List[Citation] = []

    for chunk, score in reranked[:request.top_k]:
        snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        citations.append(Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section=chunk.metadata.get("section", "General"),
            classification=chunk.classification,
            content_snippet=snippet,
            relevance_score=score
        ))

    primary_chunk = top_chunks[0]
    sec = primary_chunk.metadata.get("section", "General")
    doc_title = primary_chunk.metadata.get("document_title", primary_chunk.document_id)
    answer = f"According to {doc_title} (Section: {sec}), {primary_chunk.content.strip()}"

    # 7. Store in Bounded LRU Cache
    if request.enable_cache:
        cits_dict = [c.model_dump() for c in citations]
        semantic_cache.store(sanitized_query, query_vec, answer, cits_dict)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return QueryResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        cached=False,
        latency_ms=round(elapsed_ms, 2),
        sanitized=was_sanitized,
        detected_pii=detected_pii
    )

@router.post("/eval", response_model=EvalResponse)
def evaluate_metrics(req: EvalRequest) -> EvalResponse:
    """Evaluate retrieval and generation quality against ground truth."""
    results = evaluator.evaluate_query(
        retrieved_ids=req.retrieved_ids,
        ground_truth_ids=req.ground_truth_ids,
        answer=req.answer,
        context_chunks=req.context_chunks
    )
    return EvalResponse(**results)
