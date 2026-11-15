"""Retrieval quality on a public benchmark: BEIR SciFact (5,183 abstracts, 300 test queries).

Compares the retrieval components of this repository on the same corpus:
  bm25            repo BM25Index
  dense-hash      DenseVectorStore with the default DeterministicHashingEmbedding (128-d)
  hybrid-hash     HybridRetriever (BM25 + dense-hash, RRF)
  hybrid-hash+rr  hybrid, re-ranked by the lexical heuristic reranker in src/reranker
  dense-neural    DenseVectorStore with all-MiniLM-L6-v2 (needs sentence-transformers)
  hybrid-neural   HybridRetriever (BM25 + dense-neural, RRF)
  hybrid-neural+rr
Reports nDCG@10, MRR@10, Recall@100 and per-query latency (CPU, single thread, one query at a time).

    python scripts/evaluate_beir.py [--no-neural]      # downloads SciFact (3 MB) to data/beir/
Writes results/beir_scifact.json and results/beir_scifact.md.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.chunker import TextChunk  # noqa: E402
from src.reranker.cross_encoder import CrossEncoderReranker  # noqa: E402
from src.retrieval.embeddings import DeterministicHashingEmbedding, SentenceTransformerEmbedding  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import BM25Index  # noqa: E402
from src.retrieval.vector_store import DenseVectorStore  # noqa: E402

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA = ROOT / "data" / "beir"


def load_scifact():
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / "scifact.zip"
    if not (DATA / "scifact" / "corpus.jsonl").exists():
        urllib.request.urlretrieve(URL, zpath)
        zipfile.ZipFile(zpath).extractall(DATA)
    base = DATA / "scifact"
    corpus = {}
    for line in (base / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        corpus[d["_id"]] = (d.get("title", "") + ". " + d.get("text", "")).strip()
    queries = {}
    for line in (base / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        queries[d["_id"]] = d["text"]
    qrels: dict[str, dict[str, int]] = {}
    for row in (base / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        qid, did, score = row.split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, {})[did] = int(score)
    return corpus, {q: queries[q] for q in qrels}, qrels


def ndcg_at_k(ranked, rel, k=10):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(ranked, rel, k=10):
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked, rel, k=100):
    return len({d for d in ranked[:k] if rel.get(d, 0) > 0}) / len(rel)


def evaluate(name, search, queries, qrels):
    nd, mr, rc, lat = [], [], [], []
    for qid, text in queries.items():
        t = time.perf_counter()
        ranked = search(text)
        lat.append((time.perf_counter() - t) * 1000)
        nd.append(ndcg_at_k(ranked, qrels[qid]))
        mr.append(mrr_at_k(ranked, qrels[qid]))
        rc.append(recall_at_k(ranked, qrels[qid]))
    lat.sort()
    return {
        "config": name,
        "ndcg@10": round(statistics.mean(nd), 4),
        "mrr@10": round(statistics.mean(mr), 4),
        "recall@100": round(statistics.mean(rc), 4),
        "latency_ms_p50": round(statistics.median(lat), 2),
        "latency_ms_p95": round(lat[int(0.95 * len(lat)) - 1], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use the first N queries only")
    args = ap.parse_args()
    corpus, queries, qrels = load_scifact()
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    chunks = [
        TextChunk(chunk_id=i, document_id=i, content=t, chunk_index=0, token_count=len(t.split()))
        for i, t in corpus.items()
    ]
    print(f"corpus {len(chunks)} docs, {len(queries)} test queries", flush=True)
    bm25 = BM25Index()
    bm25.index_chunks(chunks)
    rows = [evaluate("bm25", lambda q: [c for c, _ in bm25.search(q, top_k=100)], queries, qrels)]
    print(rows[-1], flush=True)
    engines = [("hash", DeterministicHashingEmbedding(dimension=128))]
    neural_note = "not run (--no-neural)"
    if not args.no_neural:
        eng = SentenceTransformerEmbedding()
        if eng._model is not None:
            engines.append(("neural", eng))
            neural_note = "all-MiniLM-L6-v2 via sentence-transformers"
        else:
            neural_note = "sentence-transformers not installed"
    rerank = CrossEncoderReranker(top_k=100)
    for tag, eng in engines:
        vs = DenseVectorStore(embedding_engine=eng)
        t = time.perf_counter()
        vs.index_chunks(chunks)
        print(f"indexed dense-{tag} in {time.perf_counter() - t:.0f}s", flush=True)
        hyb = HybridRetriever(bm25, vs)
        rows.append(evaluate(f"dense-{tag}", lambda q: [c for c, _ in vs.search(q, top_k=100)], queries, qrels))
        print(rows[-1], flush=True)
        rows.append(evaluate(f"hybrid-{tag}", lambda q: [c.chunk_id for c, _ in hyb.search(q, top_k=100)], queries, qrels))
        print(rows[-1], flush=True)
        rows.append(
            evaluate(
                f"hybrid-{tag}+heuristic-rerank",
                lambda q: [c.chunk_id for c, _ in rerank.rerank(q, hyb.search(q, top_k=100))],
                queries,
                qrels,
            )
        )
        print(rows[-1], flush=True)
    meta = {
        "dataset": "BEIR SciFact (test)",
        "docs": len(chunks),
        "queries": len(queries),
        "neural": neural_note,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    (ROOT / "results" / "beir_scifact.json").write_text(json.dumps({"meta": meta, "results": rows}, indent=1), encoding="utf-8")
    lines = [
        "# BEIR SciFact retrieval benchmark",
        "",
        f"{meta['docs']} abstracts, {meta['queries']} test queries; neural embeddings: {meta['neural']}.",
        f"Latency: one query at a time on CPU ({meta['platform']}, Python {meta['python']}).",
        "",
        "| Config | nDCG@10 | MRR@10 | Recall@100 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['config']} | {r['ndcg@10']:.3f} | {r['mrr@10']:.3f} | {r['recall@100']:.3f} | {r['latency_ms_p50']} | {r['latency_ms_p95']} |"
            for r in rows
        ],
    ]
    (ROOT / "results" / "beir_scifact.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
