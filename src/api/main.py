"""FastAPI application entrypoint with lifespan event initialization."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, ingest_documents
from src.api.schemas import IngestRequest, DocumentPayload
from src.core.config import settings
from src.core.logging import get_logger
from src.ingestion.parser import DocumentParser

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load sample enterprise documents on startup to ensure zero-friction demonstration."""
    logger.info("Initializing Enterprise RAG service [%s]...", settings.app_env)
    
    # Auto-index sample documents if available
    sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample_documents"
    if sample_dir.exists():
        docs_to_ingest = []
        for file_path in sample_dir.glob("*.md"):
            try:
                parsed = DocumentParser.parse_file(file_path)
                docs_to_ingest.append(DocumentPayload(
                    id=parsed.id,
                    title=parsed.title,
                    content=parsed.content,
                    metadata=parsed.metadata
                ))
            except Exception as e:
                logger.error("Failed to load sample doc %s: %e", file_path.name, e)

        if docs_to_ingest:
            ingest_documents(IngestRequest(documents=docs_to_ingest))
            logger.info("Pre-indexed %d enterprise sample documents", len(docs_to_ingest))

    yield
    logger.info("Shutting down Enterprise RAG service.")

app = FastAPI(
    title="Enterprise Production RAG Platform",
    description="Enterprise-grade hybrid retrieval-augmented generation engine with semantic caching, guardrails, and cross-encoder reranking.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
