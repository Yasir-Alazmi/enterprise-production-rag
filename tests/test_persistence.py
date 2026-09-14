"""Unit tests for atomic disk persistence and checksum validation."""

import json
from pathlib import Path

from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index
from src.retrieval.vector_store import DenseVectorStore


def test_vector_store_persistence_and_checksum(tmp_path: Path):
    store = DenseVectorStore(dimension=128)
    chunks = [
        TextChunk(
            chunk_id="persist_01", document_id="doc1",
            content="Sample persistent data chunk for testing durability.",
            chunk_index=0, token_count=8, classification="INTERNAL"
        )
    ]
    store.index_chunks(chunks)

    # 1. Save to temporary disk
    saved_file = store.save_to_disk(tmp_path)
    assert saved_file.exists()

    # 2. Verify load into brand new store
    new_store = DenseVectorStore(dimension=128)
    assert len(new_store.vectors) == 0
    success = new_store.load_from_disk(tmp_path)
    assert success is True
    assert len(new_store.vectors) == 1
    assert "persist_01" in new_store.chunks_map

    # 3. Test corruption detection
    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Corrupt payload
    data["payload"]["dimension"] = 999
    with open(saved_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    corrupted_store = DenseVectorStore(dimension=128)
    assert corrupted_store.load_from_disk(tmp_path) is False

def test_bm25_persistence(tmp_path: Path):
    index = BM25Index()
    chunks = [
        TextChunk(
            chunk_id="bm_01", document_id="doc1",
            content="Disaster recovery replication across availability zones.",
            chunk_index=0, token_count=7
        )
    ]
    index.index_chunks(chunks)
    index.save_to_disk(tmp_path)

    new_index = BM25Index()
    success = new_index.load_from_disk(tmp_path)
    assert success is True
    assert "bm_01" in new_index.chunks_map
    res = new_index.search("Disaster recovery")
    assert len(res) > 0

def test_empty_store_persistence(tmp_path: Path):
    store = DenseVectorStore(dimension=128)
    saved_file = store.save_to_disk(tmp_path)
    assert saved_file.exists()

    new_store = DenseVectorStore(dimension=128)
    assert new_store.load_from_disk(tmp_path) is True
    assert len(new_store.vectors) == 0
