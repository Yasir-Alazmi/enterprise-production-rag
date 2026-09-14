"""Dense vector store with disk persistence, RBAC filtering, and idempotent deletion."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.logging import get_logger
from src.core.security import AccessControlManager
from src.ingestion.chunker import TextChunk
from src.retrieval.sparse_search import BM25Index

logger = get_logger(__name__)

class DenseVectorStore:
    """In-memory vector store with atomic disk persistence, ACL filtering, and deletion."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.chunks_map: Dict[str, TextChunk] = {}

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic, normalized pseudo-semantic feature projection."""
        tokens = BM25Index.tokenize(text)
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not tokens:
            return vec

        for idx, token in enumerate(tokens):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            pos = h % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            weight = 1.0 / (1.0 + 0.05 * idx)
            vec[pos] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks associated with a document_id. Returns number of removed chunks."""
        chunks_to_remove = [cid for cid, c in self.chunks_map.items() if c.document_id == document_id]
        for cid in chunks_to_remove:
            self.chunks_map.pop(cid, None)
            self.vectors.pop(cid, None)
        if chunks_to_remove:
            logger.info("Removed %d chunks for document '%s' from vector store", len(chunks_to_remove), document_id)
        return len(chunks_to_remove)

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Generate embeddings and index candidate chunks with deduplication."""
        # Clean existing chunks for documents being re-indexed (idempotent upsert)
        doc_ids_to_update = {chunk.document_id for chunk in chunks}
        for doc_id in doc_ids_to_update:
            self.delete_document(doc_id)

        for chunk in chunks:
            self.chunks_map[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = self._embed(chunk.content)

    def search(
        self,
        query: str,
        top_k: int = 10,
        user_role: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Compute cosine similarity filtered by user clearance."""
        if not self.vectors:
            return []

        query_vec = self._embed(query)
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-6:
            return []

        scores: List[Tuple[str, float]] = []
        for chunk_id, doc_vec in self.vectors.items():
            chunk = self.chunks_map.get(chunk_id)
            if user_role and chunk:
                if not AccessControlManager.can_access(user_role, chunk.classification):
                    continue

            sim = float(np.dot(query_vec, doc_vec))
            scores.append((chunk_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save_to_disk(self, storage_dir: Path) -> Path:
        """Atomically persist vectors, chunks, and integrity checksum to disk."""
        storage_dir.mkdir(parents=True, exist_ok=True)
        target_file = storage_dir / "vector_store.json"

        serialized_chunks = {cid: c.model_dump() for cid, c in self.chunks_map.items()}
        serialized_vectors = {cid: vec.tolist() for cid, vec in self.vectors.items()}

        payload = {
            "dimension": self.dimension,
            "chunks": serialized_chunks,
            "vectors": serialized_vectors
        }

        raw_json = json.dumps(payload, indent=2)
        checksum = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()

        manifest = {
            "checksum_sha256": checksum,
            "total_vectors": len(self.vectors),
            "payload": payload
        }

        tmp_file = storage_dir / "vector_store.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        tmp_file.replace(target_file)

        return target_file

    def load_from_disk(self, storage_dir: Path) -> bool:
        """Load and verify persisted vectors from disk."""
        target_file = storage_dir / "vector_store.json"
        if not target_file.exists():
            return False

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            payload = manifest.get("payload", {})
            recorded_checksum = manifest.get("checksum_sha256")

            raw_json = json.dumps(payload, indent=2)
            calculated_checksum = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
            if recorded_checksum and recorded_checksum != calculated_checksum:
                logger.error("Vector store checksum mismatch: corrupted snapshot!")
                return False

            self.dimension = payload.get("dimension", 128)
            self.chunks_map = {cid: TextChunk(**data) for cid, data in payload.get("chunks", {}).items()}
            self.vectors = {cid: np.array(vec, dtype=np.float32) for cid, vec in payload.get("vectors", {}).items()}
            return True
        except Exception as e:
            logger.error("Failed to load vector store snapshot: %s", e)
            return False
