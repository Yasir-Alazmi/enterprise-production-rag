"""BM25 Okapi sparse search with RBAC filtering and disk persistence."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.core.security import AccessControlManager
from src.ingestion.chunker import TextChunk


class BM25Index:
    """Inverted index implementing BM25Okapi scoring with ACL filtering."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.term_doc_freq: Dict[str, int] = Counter()
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.chunks_map: Dict[str, TextChunk] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normalize and tokenize text into lowercase alphanumeric terms."""
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Build or update the inverted index with provided chunks."""
        for chunk in chunks:
            self.chunks_map[chunk.chunk_id] = chunk
            tokens = self.tokenize(chunk.content)
            length = len(tokens)
            self.doc_len[chunk.chunk_id] = length

            term_counts = Counter(tokens)
            unique_terms: Set[str] = set(term_counts.keys())
            for term in unique_terms:
                self.term_doc_freq[term] += 1

            for term, freq in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][chunk.chunk_id] = freq

        self.doc_count = len(self.doc_len)
        if self.doc_count > 0:
            self.avg_doc_len = sum(self.doc_len.values()) / self.doc_count

    def search(
        self,
        query: str,
        top_k: int = 10,
        user_role: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Compute BM25 scores filtered by user clearance."""
        query_tokens = self.tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []

        scores: Dict[str, float] = Counter()

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            df = self.term_doc_freq[term]
            idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

            for chunk_id, tf in self.inverted_index[term].items():
                chunk = self.chunks_map.get(chunk_id)
                if user_role and chunk:
                    if not AccessControlManager.can_access(user_role, chunk.classification):
                        continue

                doc_len = self.doc_len.get(chunk_id, self.avg_doc_len)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                score = idf * ((tf * (self.k1 + 1.0)) / max(1e-6, denom))
                scores[chunk_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save_to_disk(self, storage_dir: Path) -> Path:
        """Persist BM25 inverted index state to disk."""
        storage_dir.mkdir(parents=True, exist_ok=True)
        target = storage_dir / "bm25_index.json"
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_len": self.doc_len,
            "avg_doc_len": self.avg_doc_len,
            "doc_count": self.doc_count,
            "term_doc_freq": dict(self.term_doc_freq),
            "inverted_index": self.inverted_index,
            "chunks": {cid: c.model_dump() for cid, c in self.chunks_map.items()}
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return target

    def load_from_disk(self, storage_dir: Path) -> bool:
        """Restore BM25 inverted index state from disk."""
        target = storage_dir / "bm25_index.json"
        if not target.exists():
            return False
        try:
            with open(target, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.k1 = payload.get("k1", 1.5)
            self.b = payload.get("b", 0.75)
            self.doc_len = payload.get("doc_len", {})
            self.avg_doc_len = payload.get("avg_doc_len", 0.0)
            self.doc_count = payload.get("doc_count", 0)
            self.term_doc_freq = Counter(payload.get("term_doc_freq", {}))
            self.inverted_index = payload.get("inverted_index", {})
            self.chunks_map = {cid: TextChunk(**data) for cid, data in payload.get("chunks", {}).items()}
            return True
        except Exception:
            return False
