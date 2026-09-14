"""Pydantic v2 schemas for API contracts including RBAC and metrics."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document text content")
    classification: Optional[str] = Field(default="INTERNAL", description="PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")

class IngestRequest(BaseModel):
    documents: List[DocumentPayload] = Field(..., description="List of documents to ingest")
    persist_to_disk: Optional[bool] = Field(default=True, description="Whether to serialize index to storage")

class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    persisted: bool
    status: str = "success"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User search query or question")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of top reranked results to return")
    user_role: Optional[str] = Field(default="employee", description="Role for RBAC filtering (guest, employee, manager, admin)")
    enable_cache: Optional[bool] = Field(default=True, description="Whether to check semantic cache")

class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_title: Optional[str] = None
    section: str
    classification: str
    content_snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    cached: bool
    latency_ms: float
    retrieval_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    sanitized: bool
    detected_pii: List[str] = Field(default_factory=list)

class EvalRequest(BaseModel):
    retrieved_ids: List[str]
    ground_truth_ids: List[str]
    answer: str
    context_chunks: List[str]
    query: Optional[str] = None

class EvalResponse(BaseModel):
    context_precision: float
    context_recall: float
    mrr: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    faithfulness: float
    answer_relevance: Optional[float] = None

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.4.0"
    indexed_chunks: int
    cache_entries: int
    cache_stats: Dict[str, Any]
