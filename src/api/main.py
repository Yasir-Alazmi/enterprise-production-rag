"""FastAPI application entrypoint with lifespan event initialization and production middleware."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import SlidingWindowRateLimiter, TelemetryMiddleware
from src.api.routes import STORAGE_DIR, bm25_index, ingest_documents, router, vector_store
from src.api.schemas import DocumentPayload, IngestRequest
from src.core.config import settings
from src.core.logging import get_logger
from src.ingestion.parser import DocumentParser

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Restore persistent disk snapshot or load sample documents on boot."""
    logger.info("Initializing Enterprise RAG service [%s]...", settings.app_env)

    # Check if disk persistence exists
    restored_vec = vector_store.load_from_disk(STORAGE_DIR)
    restored_bm25 = bm25_index.load_from_disk(STORAGE_DIR)

    if restored_vec and restored_bm25:
        logger.info("Successfully restored indexed data from persistent disk storage.")
    else:
        logger.info("No existing persistent snapshot found. Pre-indexing sample enterprise documents...")
        sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample_documents"
        if sample_dir.exists():
            docs_to_ingest = []
            for file_path in sample_dir.glob("*.md"):
                try:
                    # Classify documents based on title
                    classification = "CONFIDENTIAL" if "security" in file_path.name else "INTERNAL"
                    parsed = DocumentParser.parse_file(file_path, classification=classification)
                    docs_to_ingest.append(DocumentPayload(
                        id=parsed.id,
                        title=parsed.title,
                        content=parsed.content,
                        classification=parsed.classification,
                        metadata=parsed.metadata
                    ))
                except Exception as e:
                    logger.error("Failed to load sample doc %s: %s", file_path.name, e)

            if docs_to_ingest:
                ingest_documents(IngestRequest(documents=docs_to_ingest, persist_to_disk=True))
                logger.info("Pre-indexed and persisted %d sample documents", len(docs_to_ingest))

    yield
    logger.info("Flushing state and gracefully shutting down Enterprise RAG service.")
    vector_store.save_to_disk(STORAGE_DIR)
    bm25_index.save_to_disk(STORAGE_DIR)

app = FastAPI(
    title="Enterprise Production RAG Platform",
    description="Enterprise-grade hybrid retrieval-augmented generation engine with persistent storage, RBAC, semantic caching, and Prometheus observability.",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TelemetryMiddleware)
app.add_middleware(SlidingWindowRateLimiter, max_requests_per_minute=200)

app.include_router(router)
