"""Unit tests for document parsing and recursive chunking."""

import pytest

from src.core.exceptions import ChunkingError
from src.ingestion.chunker import RecursiveTokenChunker
from src.ingestion.parser import Document, DocumentParser


def test_chunker_invalid_parameters():
    with pytest.raises(ChunkingError):
        RecursiveTokenChunker(chunk_size=0)
    with pytest.raises(ChunkingError):
        RecursiveTokenChunker(chunk_size=100, chunk_overlap=150)

def test_chunker_splits_document_correctly(sample_document: Document):
    chunker = RecursiveTokenChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.split_document(sample_document)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.document_id == sample_document.id
        assert chunk.token_count > 0
        assert len(chunk.content.strip()) > 0
        assert "document_title" in chunk.metadata

def test_chunker_empty_document():
    chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=20)
    empty_doc = Document(id="empty", title="Empty", content="")
    chunks = chunker.split_document(empty_doc)
    assert chunks == []

def test_parser_memory_string():
    doc = DocumentParser.parse_text("Sample body", doc_id="doc_test")
    assert doc.id == "doc_test"
    assert doc.title == "Doc Test"
    assert doc.metadata["byte_size"] > 0
