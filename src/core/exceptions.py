"""Custom domain exceptions for enterprise RAG pipeline."""

class RAGException(Exception):
    """Base exception for all RAG engine errors."""
    pass

class DocumentParsingError(RAGException):
    """Raised when document ingestion or parsing fails."""
    pass

class ChunkingError(RAGException):
    """Raised when text segmentation encounters invalid inputs or tokens."""
    pass

class RetrievalError(RAGException):
    """Raised when vector or sparse retrieval fails."""
    pass

class SecurityViolationError(RAGException):
    """Raised when prompt injection or unpermitted input is detected."""
    pass

class ConfigurationError(RAGException):
    """Raised when settings or configuration parameters are invalid."""
    pass
