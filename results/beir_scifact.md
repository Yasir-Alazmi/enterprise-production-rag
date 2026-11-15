# BEIR SciFact retrieval benchmark

5183 abstracts, 300 test queries; neural embeddings: all-MiniLM-L6-v2 via sentence-transformers.
Latency: one query at a time on CPU (Windows-11-10.0.26200-SP0, Python 3.12.9).

| Config | nDCG@10 | MRR@10 | Recall@100 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.661 | 0.628 | 0.876 | 13.84 | 32.35 |
| dense-hash | 0.172 | 0.152 | 0.439 | 9.02 | 10.9 |
| hybrid-hash | 0.346 | 0.320 | 0.844 | 25.3 | 45.5 |
| hybrid-hash+heuristic-rerank | 0.491 | 0.456 | 0.844 | 44.8 | 74.5 |
| dense-neural | 0.648 | 0.607 | 0.925 | 22.32 | 33.91 |
| hybrid-neural | 0.679 | 0.638 | 0.955 | 44.81 | 63.58 |
| hybrid-neural+heuristic-rerank | 0.647 | 0.611 | 0.955 | 51.68 | 82.74 |
