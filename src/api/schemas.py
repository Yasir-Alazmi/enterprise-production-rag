"""Pydantic v2 schemas for API contracts."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")

class IngestRequest(BaseModel):
    documents: List[DocumentPayload] = Field(..., description="List of documents to ingest")

class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    status: str = "success"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User search query or question")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of top reranked results to return")
    enable_cache: Optional[bool] = Field(default=True, description="Whether to check semantic cache")

class Citation(BaseModel):
    chunk_id: str
    document_id: str
    section: str
    content_snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    cached: bool
    latency_ms: float
    sanitized: bool
    detected_pii: List[str] = Field(default_factory=list)

class EvalRequest(BaseModel):
    retrieved_ids: List[str]
    ground_truth_ids: List[str]
    answer: str
    context_chunks: List[str]

class EvalResponse(BaseModel):
    context_precision: float
    context_recall: float
    faithfulness: float

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    indexed_chunks: int
    cache_entries: int
    cache_stats: Dict[str, Any]
