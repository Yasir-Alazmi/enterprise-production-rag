"""Document parser supporting Markdown, structured text, and security classifications."""

import re
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.core.exceptions import DocumentParsingError
from src.core.logging import get_logger

logger = get_logger(__name__)

class Document(BaseModel):
    """Represents a raw ingested document with security classification and metadata."""
    id: str
    title: str
    content: str
    classification: str = Field(default="INTERNAL", description="PUBLIC, INTERNAL, CONFIDENTIAL, or RESTRICTED")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentParser:
    """Extracts clean text and structural sections from documents."""

    @staticmethod
    def parse_file(file_path: Path, classification: str = "INTERNAL") -> Document:
        """Parse a local text or markdown file into a Document instance."""
        if not file_path.exists():
            raise DocumentParsingError(f"Document file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception as e:
            raise DocumentParsingError(f"Failed to read file {file_path}: {e}") from e

        title = file_path.stem.replace("_", " ").title()

        # Check if first header contains a markdown title
        header_match = re.search(r"^#\s+(.+)$", raw_text, re.MULTILINE)
        if header_match:
            title = header_match.group(1).strip()

        metadata = {
            "source": str(file_path.name),
            "file_type": file_path.suffix.lower(),
            "byte_size": len(raw_text.encode("utf-8")),
            "classification": classification.upper().strip()
        }

        logger.info("Parsed document: %s [%s] (%d bytes)", title, metadata["classification"], metadata["byte_size"])
        return Document(
            id=file_path.stem,
            title=title,
            content=raw_text,
            classification=classification.upper().strip(),
            metadata=metadata
        )

    @staticmethod
    def parse_text(
        text: str,
        doc_id: str,
        title: Optional[str] = None,
        classification: str = "INTERNAL",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a Document instance directly from an in-memory string."""
        if not text.strip():
            raise DocumentParsingError("Document text cannot be empty.")

        doc_title = title or doc_id.replace("_", " ").title()
        meta = metadata or {}
        meta.setdefault("source", "in_memory")
        meta.setdefault("byte_size", len(text.encode("utf-8")))
        meta["classification"] = classification.upper().strip()

        return Document(
            id=doc_id,
            title=doc_title,
            content=text,
            classification=classification.upper().strip(),
            metadata=meta
        )
