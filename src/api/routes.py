"""API Route definitions and Generative RAG pipeline execution with Bearer Auth."""

import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, Response, status

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
from src.core.security import AccessControlManager, ClassificationLevel
from src.evaluation.metrics import RAGEvaluator
from src.generation.generator import get_answer_generator
from src.guardrails.injection_detector import InjectionDetector
from src.guardrails.pii_sanitizer import PIISanitizer
from src.ingestion.chunker import RecursiveTokenChunker, TextChunk
from src.ingestion.parser import Document
from src.reranker.cross_encoder import CrossEncoderReranker
from src.retrieval.embeddings import get_embedding_engine
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "storage"

# Global pipeline instances with pluggable embedding and generation engines
chunker = RecursiveTokenChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
bm25_index = BM25Index(k1=settings.sparse_k1, b=settings.sparse_b)
embedding_engine = get_embedding_engine(
    provider=settings.embedding_provider,
    dimension=settings.embedding_dimension
)
vector_store = DenseVectorStore(
    dimension=embedding_engine.dimension,
    embedding_engine=embedding_engine
)
hybrid_retriever = HybridRetriever(bm25_index=bm25_index, vector_store=vector_store, alpha=settings.hybrid_alpha)
reranker = CrossEncoderReranker(top_k=settings.rerank_top_k)
generator = get_answer_generator(provider=settings.generation_provider)
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
    """Expose Prometheus operational telemetry with stage latency breakdowns."""
    text_content = metrics_collector.export_text(
        cache_stats=semantic_cache.stats(),
        indexed_chunks=len(vector_store.chunks_map)
    )
    return Response(content=text_content, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    authorization: Optional[str] = Header(default=None)
) -> dict:
    """Delete all chunks for a document from vector and sparse indexes.

    Zero-Trust Security: Requires valid Bearer token with at least CONFIDENTIAL clearance.
    """
    _, clearance = AccessControlManager.resolve_bearer_identity(
        auth_header=authorization,
        require_auth=True
    )
    AccessControlManager.enforce_clearance(
        user_clearance=clearance,
        min_clearance=ClassificationLevel.CONFIDENTIAL,
        operation_name="document deletion"
    )

    v_del = vector_store.delete_document(document_id)
    b_del = bm25_index.delete_document(document_id)

    # Update persistent store
    vector_store.save_to_disk(STORAGE_DIR)
    bm25_index.save_to_disk(STORAGE_DIR)

    if v_del == 0 and b_del == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in any index."
        )

    logger.info("Deleted document '%s' (vector chunks: %d, bm25 chunks: %d)", document_id, v_del, b_del)
    return {
        "status": "success",
        "document_id": document_id,
        "deleted_chunks": max(v_del, b_del)
    }


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_documents(
    payload: IngestRequest,
    authorization: Optional[str] = Header(default=None)
) -> IngestResponse:
    """Ingest, chunk, index, and optionally persist documents with RBAC classifications.

    Zero-Trust Security: Requires valid Bearer token with at least CONFIDENTIAL clearance.
    """
    _, clearance = AccessControlManager.resolve_bearer_identity(
        auth_header=authorization,
        require_auth=True
    )
    AccessControlManager.enforce_clearance(
        user_clearance=clearance,
        min_clearance=ClassificationLevel.CONFIDENTIAL,
        operation_name="document ingestion"
    )

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
def query_pipeline(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None)
) -> QueryResponse:
    """Execute end-to-end Generative RAG with Bearer Auth, RBAC, and role-scoped caching."""
    start_time = time.perf_counter()

    # 1. Bearer Token Identity Resolution (Zero-trust: header is sole source of truth)
    resolved_role, clearance = AccessControlManager.resolve_bearer_identity(
        auth_header=authorization,
        require_auth=False
    )

    # 2. Adversarial Injection Screening
    if settings.enable_injection_detection:
        is_safe, violation = injection_detector.validate_prompt(request.query)
        if not is_safe:
            logger.warning("Rejected malicious prompt: %s", violation)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "SecurityViolation", "message": violation}
            )

    # 3. PII Sanitization
    sanitized_query = request.query
    detected_pii: List[str] = []
    was_sanitized = False
    if settings.enable_pii_masking:
        sanitized_query, was_sanitized, detected_pii = sanitizer.sanitize(request.query)

    # 4. Role-Scoped Semantic Cache Check
    query_vec = vector_store._embed(sanitized_query)
    if request.enable_cache:
        cached_result = semantic_cache.lookup(
            sanitized_query,
            query_vec,
            user_clearance=int(clearance.value)
        )
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
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                sanitized=was_sanitized,
                detected_pii=detected_pii
            )

    # 5. Hybrid Retrieval with RBAC filtering & Stage Latency Tracking
    t_ret_start = time.perf_counter()
    candidates = hybrid_retriever.search(
        sanitized_query,
        top_k=request.top_k * 2,
        user_role=resolved_role
    )

    if not candidates:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        ret_duration = time.perf_counter() - t_ret_start
        metrics_collector.record_stage("retrieval", ret_duration)
        return QueryResponse(
            query=request.query,
            answer="No relevant documentation matching your security clearance was found.",
            citations=[],
            cached=False,
            latency_ms=round(elapsed_ms, 2),
            retrieval_latency_ms=round(ret_duration * 1000.0, 2),
            generation_latency_ms=0.0,
            sanitized=was_sanitized,
            detected_pii=detected_pii
        )

    # 6. Cross-Encoder Reranking
    reranked = reranker.rerank(sanitized_query, candidates)
    ret_duration = time.perf_counter() - t_ret_start
    metrics_collector.record_stage("retrieval", ret_duration)

    # 7. Multi-Source Generative Answer Synthesis & Stage Latency Tracking
    t_gen_start = time.perf_counter()
    top_candidates = reranked[:request.top_k]
    gen_result = generator.generate_answer(sanitized_query, top_candidates)
    gen_duration = time.perf_counter() - t_gen_start
    metrics_collector.record_stage("generation", gen_duration)

    citations: List[Citation] = []
    for chunk, score in top_candidates:
        snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        citations.append(Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section=chunk.metadata.get("section", "General"),
            classification=chunk.classification,
            content_snippet=snippet,
            relevance_score=score
        ))

    # 8. Store in Role-Scoped Cache
    if request.enable_cache and gen_result.grounded:
        cits_dict = [c.model_dump() for c in citations]
        semantic_cache.store(
            sanitized_query,
            query_vec,
            gen_result.answer,
            cits_dict,
            clearance_level=int(clearance.value)
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return QueryResponse(
        query=request.query,
        answer=gen_result.answer,
        citations=citations,
        cached=False,
        latency_ms=round(elapsed_ms, 2),
        retrieval_latency_ms=round(ret_duration * 1000.0, 2),
        generation_latency_ms=round(gen_duration * 1000.0, 2),
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
        context_chunks=req.context_chunks,
        query=req.query or ""
    )
    return EvalResponse(**results)
