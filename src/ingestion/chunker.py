"""Recursive character and semantic token chunker."""

import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.core.exceptions import ChunkingError
from src.core.logging import get_logger
from src.ingestion.parser import Document

logger = get_logger(__name__)

class TextChunk(BaseModel):
    """Represents an atomic chunk of text prepared for indexing."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecursiveTokenChunker:
    """Segments documents recursively based on natural paragraph and sentence boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than zero.")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count based on whitespace and punctuation splitting."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def split_document(self, document: Document) -> List[TextChunk]:
        """Split a Document into bounded, overlapping TextChunks."""
        if not document.content.strip():
            return []

        # Split document by paragraphs first
        paragraphs = re.split(r"\n\s*\n", document.content)
        current_section = "General"

        raw_chunks: List[str] = []
        section_tags: List[str] = []
        current_buffer: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if paragraph is a markdown header
            header_match = re.match(r"^(#{1,4})\s+(.+)$", para)
            if header_match:
                current_section = header_match.group(2).strip()
                continue

            para_tokens = self.estimate_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_buffer:
                chunk_text = " ".join(current_buffer)
                raw_chunks.append(chunk_text)
                section_tags.append(current_section)

                # Apply overlap from end of current buffer
                overlap_buffer: List[str] = []
                overlap_tokens = 0
                for item in reversed(current_buffer):
                    item_tok = self.estimate_tokens(item)
                    if overlap_tokens + item_tok <= self.chunk_overlap:
                        overlap_buffer.insert(0, item)
                        overlap_tokens += item_tok
                    else:
                        break

                current_buffer = overlap_buffer + [para]
                current_tokens = overlap_tokens + para_tokens
            else:
                current_buffer.append(para)
                current_tokens += para_tokens

        if current_buffer:
            raw_chunks.append(" ".join(current_buffer))
            section_tags.append(current_section)

        # Build output models
        chunks: List[TextChunk] = []
        for idx, (chunk_text, section) in enumerate(zip(raw_chunks, section_tags)):
            chunk_id = f"{document.id}_c{idx:03d}"
            meta = dict(document.metadata)
            meta.update({
                "document_title": document.title,
                "section": section,
                "chunk_index": idx
            })
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                chunk_index=idx,
                token_count=self.estimate_tokens(chunk_text),
                metadata=meta
            ))

        logger.info("Split document '%s' into %d chunks", document.id, len(chunks))
        return chunks
