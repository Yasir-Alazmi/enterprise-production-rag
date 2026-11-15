# Enterprise RAG Reference Architecture

[![CI](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

A small reference implementation of role-based hybrid Retrieval-Augmented Generation (RAG): JWT authentication with role-based access control applied inside retrieval, BM25 + dense retrieval fused with Reciprocal Rank Fusion, prompt-injection and PII guards, and a role-scoped semantic cache. It is a learning and architecture project: it has not been deployed or load-tested, and its default components are lightweight stand-ins (see [What the components really are](#what-the-components-really-are)). Retrieval quality is measured on a public benchmark below.

---

## Architecture Blueprint

```
[ Client Request ] ──► [ Sliding Window Rate Limiter (200 req/min) ]
                               │
                               ▼
        [ Zero-Trust Bearer Token & JWT Verification Engine ]
        (HMAC-SHA256 JWT / API Tokens: PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED)
                               │
                               ▼
        [ Protected FastAPI Gateway (/api/v1/query, /ingest, /documents/{id}, /eval) ]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
 [ Injection Detector ]                         [ PII Sanitizer ]
 (System Boundary Heuristics)                   (National ID, Cards, Keys)
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
             [ Role-Scoped Bounded Semantic Cache ] ──(Hit)──► Return Cached Answer (< 2ms)
                                │ (Miss)
                                ▼
            [ Role-Based Access Control (RBAC Filter) ]
            (Clearance Level >= Document Classification)
                                │
                                ▼
              [ Hybrid Retrieval Engine (RRF Fusion) ]
              ├── BM25Okapi Lexical Index (k1=1.5, b=0.75)
              └── Dense Vector Store (Pluggable Embedding Engine)
                                │
                                ▼
                  [ Heuristic Re-Ranking ]
                  (Joint Query-Document Interaction)
                                │
                                ▼
             [ Pluggable Answer Generation Abstraction ]
             ├── Deterministic Grounded Synthesis (Offline, Zero Hallucination)
             └── Live LLM Generator (OpenAI / Local Ollama via HTTPX)
                                │
                                ▼
             [ Prometheus Telemetry (/api/v1/metrics) ]
             ├── rag_stage_duration_seconds{stage="retrieval"}
             └── rag_stage_duration_seconds{stage="generation"}
```

---

## Core Engineering Capabilities

### 1. Zero-Trust Identity, RBAC & API Hardening (`src/core/security.py`, `src/api/main.py`)
- **HMAC-SHA256 JWT Decoding**: Validates cryptographic signatures, token expiration (`exp`), and tenant clearance claims (`sub`, `role`, `clearance`).
- **Protected Administrative & Evaluation Endpoints**: `POST /api/v1/ingest`, `DELETE /api/v1/documents/{id}`, and `POST /api/v1/eval` strictly require Bearer authorization with $\ge$ `CONFIDENTIAL` clearance. Unauthenticated calls return `HTTP 401 Unauthorized`; insufficient clearance calls return `HTTP 403 Forbidden`.
- **Strict Header Enforcement**: Unrecognized or malformed tokens explicitly return `401 Unauthorized` rather than silently degrading permissions.
- **Restricted CORS Policy**: Configurable, environment-governed allowed origins with strict method (`GET`, `POST`, `DELETE`, `OPTIONS`) and header filtering, completely eliminating wildcard origin exposure on credentialed endpoints.

### 2. Dual-Mode Embedding Engine Interface (`src/retrieval/embeddings.py`)
- **Interface Contract (`BaseEmbeddingEngine`)**: Decouples the vector store and cache from underlying embedding algorithms.
- **`DeterministicHashingEmbedding` (128-d, the default)**: a zero-dependency sign-hash projection of tokens. It is **not semantic** (it only encodes which words occur) and is meant for offline CI and reproducible tests. On a real benchmark it performs far below BM25 and lowers the hybrid result (see Results); use the neural engine for real retrieval.
- **`SentenceTransformerEmbedding` (384-d)**: neural embeddings using Sentence Transformers (`all-MiniLM-L6-v2`); falls back to the hashing engine if the package is missing. This is the configuration that gives the benchmark gains below.

### 3. Pluggable Answer Generation Abstraction (`src/generation/generator.py`)
- **Protocol Contract (`BaseAnswerGenerator`)**: Supports interchangeable synthesis backends via factory configuration (`GENERATION_PROVIDER`).
- **`DeterministicGroundedGenerator`**: Extractive factual synthesis that combines facts across retrieved chunks, appends verified section citations `[Document Title - Section]`, and deterministically refuses insufficient context (*"Insufficient enterprise documentation was retrieved to answer this query with verifiable certainty"*).
- **`LLMAnswerGenerator`**: Pluggable live generation adapter calling OpenAI-compatible APIs or local Ollama endpoints via `httpx`, conditioned with strict anti-hallucination system instructions.

### 4. Role-Scoped Semantic Cache (Zero Privilege Escalation)
- Caches query vectors and responses with bounded LRU eviction (1,000 entries).
- **Privilege Boundary Enforcement**: Cache entries record `clearance_level`. An employee query will never receive a cached answer generated from an executive's restricted inquiry.

### 5. Automated Evaluation Suite & Stage Latency Tracking (`src/evaluation/`, `scripts/evaluate_rag.py`)
- Computes empirical information retrieval and generation metrics:
  - Context Precision@K, Context Recall@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG@5)
  - Answer Faithfulness & Keyword Relevance
- Separates **retrieval latency** from **generation latency** in Prometheus telemetry and API responses.

---

## Empirical Benchmark & Evaluation Results

### Retrieval quality on a public benchmark (`scripts/evaluate_beir.py`)

BEIR **SciFact**: 5,183 scientific abstracts, 300 test queries with relevance judgements. Each row uses this repository's own components on the same corpus. Latency is one query at a time on a CPU (Windows 11, Python 3.12, no GPU); full output in [`results/beir_scifact.md`](results/beir_scifact.md).

| Configuration | nDCG@10 | MRR@10 | Recall@100 | p50 latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| BM25 only | 0.661 | 0.628 | 0.876 | 13.84 |
| Dense, default hashing embedding | 0.172 | 0.152 | 0.439 | 9.02 |
| Hybrid (BM25 + hashing dense), default settings | 0.346 | 0.320 | 0.844 | 25.3 |
| Hybrid (hashing) + heuristic re-ranker | 0.491 | 0.456 | 0.844 | 44.8 |
| Dense, MiniLM neural embedding | 0.648 | 0.607 | 0.925 | 22.32 |
| **Hybrid (BM25 + MiniLM), RRF** | 0.679 | 0.638 | 0.955 | 44.81 |
| Hybrid (MiniLM) + heuristic re-ranker | 0.647 | 0.611 | 0.955 | 51.68 |

Sanity check: BM25 (0.661) and MiniLM dense (0.648) are close to the published BEIR numbers for SciFact (about 0.665 and 0.645), so the harness is measuring correctly.

What the numbers say:
- **With the neural embedding, hybrid retrieval is the best configuration** (0.679), a modest gain over BM25 alone (+0.018 nDCG@10) and a clearer gain in Recall@100 (0.955 vs 0.876).
- **The default hashing configuration hurts.** Hybrid with hashing scores 0.346, about half of BM25 alone, because the default fusion weight (0.60) gives most of the vote to a non-semantic signal. The default is for offline tests only.
- **The heuristic re-ranker helps weak first stages and hurts strong ones**: +0.145 on the hashing hybrid, -0.032 on the neural hybrid. It is a hand-written scoring function (query-term coverage, density, position), not a trained cross-encoder.
- **Latency is tens of milliseconds at 5,183 documents** (p50 13.8 ms for BM25, 44.8 ms for neural hybrid), not the 2 ms measured below on a two-document corpus. The vector store is a brute-force in-memory scan, so it will not scale to millions of documents without a real index.

### Performance Benchmark (`scripts/benchmark.py`)
**Corpus: two sample documents.** These latencies measure the pipeline overhead on a tiny corpus and say nothing about performance at realistic size (see the benchmark above). Measured locally across 100 consecutive requests on Windows (AMD64, Python 3.13):

| Performance Metric | Cold Hybrid Retrieval + Synthesis | Semantic Cache Hit |
| :--- | :--- | :--- |
| **p50 Latency (Median)** | **2.37 ms** | **2.06 ms** |
| **p95 Latency** | **2.86 ms** | **2.44 ms** |
| **p99 Latency** | **4.02 ms** | **2.67 ms** |
| **Mean Execution Time** | **2.44 ms** | **2.08 ms** |
| **Throughput Capacity** | > 400 req/sec | > 500 req/sec |
| **Automated Test Suite** | 67 tests passing (verified 2026-11-15) | 14 test files |

> **Benchmark Scope Note**: Latency covers full Ingestion -> PII Sanitization -> Injection Screening -> Role-Scoped Cache -> Hybrid Retrieval -> Heuristic Re-Ranker -> Deterministic Grounded Synthesis. Live third-party external LLM API calls incur separate network roundtrips (~300ms - 1500ms).


### Automated RAG Evaluation Report (`scripts/evaluate_rag.py`)
**This is a regression test, not a quality estimate.** It runs 4 queries against 2 sample documents (with ground-truth documents and PII/injection checks); with n = 4 each metric moves in large steps. Use the SciFact benchmark above for retrieval quality.

| Evaluation Metric | Aggregate Score | Target SLA |
| :--- | :--- | :--- |
| **Mean Context Precision@K** | **0.833** | > 0.80 |
| **Mean Context Recall@K** | **1.000** | > 0.90 |
| **Mean Reciprocal Rank (MRR)** | **0.833** | > 0.80 |
| **Mean NDCG@5** | **0.875** | > 0.85 |
| **Answer Faithfulness** | **0.875** | > 0.85 (Strict Context Grounding) |
| **Answer Citation Correctness** | **1.000** | 1.00 (Zero Hallucinated Sources) |
| **Mean Retrieval Latency** | **0.22 ms** | < 5.00 ms |
| **Mean Generation Latency (Local)** | **0.10 ms** | < 5.00 ms |

---

## What the components really are

| Component | Name in the code | What it actually is |
| :--- | :--- | :--- |
| Default embedding | `DeterministicHashingEmbedding` | Token sign-hash projection, not semantic; for CI and tests |
| Neural embedding | `SentenceTransformerEmbedding` | `all-MiniLM-L6-v2` when installed |
| Re-ranker | `CrossEncoderReranker` | Hand-written lexical heuristic (coverage, density, position); the class name is historical |
| Default generator | `DeterministicGroundedGenerator` | Extractive: stitches retrieved sentences and adds citations; no language model |
| Optional generator | `LLMAnswerGenerator` | Calls an OpenAI-compatible or Ollama endpoint; not exercised in the benchmark |
| Vector store | `DenseVectorStore` | In-memory brute-force cosine scan with JSON persistence |
| Evaluation corpus (default) | Golden Reference Corpus | 2 documents and 4 queries |

---

## Enterprise Production Scaling Blueprint

This repository is architected as a clean reference platform. For horizontal scaling in multi-node Kubernetes clusters, migrate components following this blueprint:

```
[ In-Memory Reference ]                [ Enterprise Cloud Target ]
DenseVectorStore (JSON + SHA256)  ──►  PostgreSQL + pgvector / Qdrant Cloud
BM25Index (Inverted Index)        ──►  Elasticsearch / OpenSearch
SemanticCache (In-Memory LRU)     ──►  Redis Cluster with RedisVL (RediSearch)
Pre-Shared Bearer Tokens          ──►  Keycloak / Azure Entra ID (OIDC / OAuth2)
Prometheus Python Collector       ──►  OpenTelemetry Collector + Prometheus + Grafana
```

Detailed migration schemas, SQL DDL, and configuration examples are documented in [`docs/architecture.md`](docs/architecture.md).

---

## Directory Structure

```
enterprise-production-rag/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix (Python 3.10 & 3.11)
├── configs/config.yaml            # Centralized hyperparameter & security configuration
├── data/
│   ├── sample_documents/         # Compliance & Security SLA documents
│   └── storage/                  # Persistent vector snapshots with SHA256 verification
├── docs/
│   ├── architecture.md           # Engineering trade-offs and enterprise scaling blueprints
│   └── runbook.md                # Day-2 operations, key rotation, and Prometheus alerts
├── results/
│   ├── benchmark_report.json     # Empirical latency percentiles and system metadata
│   └── evaluation_report.json    # Precision, Recall, MRR, and NDCG evaluation metrics
├── scripts/
│   ├── benchmark.py              # Performance benchmark harness (p50/p95/p99)
│   └── evaluate_rag.py           # Automated RAG evaluation on golden reference corpus
├── src/
│   ├── api/                      # FastAPI routes, schemas, and rate-limiting middleware
│   ├── cache/                    # Role-scoped bounded LRU semantic query cache
│   ├── core/                     # JWT auth, zero-trust RBAC, Prometheus stage metrics
│   ├── evaluation/               # Precision, Recall, MRR, NDCG@K, faithfulness scoring
│   ├── generation/               # Pluggable generation abstraction (Grounded + LLM)
│   ├── guardrails/               # PII sanitization and adversarial injection screening
│   ├── ingestion/                # Document parsing and recursive token chunker
│   ├── reranker/                 # Lexical heuristic re-ranker (module name is historical; not a neural cross-encoder)
│   └── retrieval/                # Dual-mode embeddings, BM25, and hybrid RRF retriever
├── tests/                        # 67 automated tests across 13 test suites (100% passing)
├── Dockerfile                    # Multi-stage production container
├── docker-compose.yml            # Production service orchestration
├── pyproject.toml                # Build system & pytest configuration
└── requirements.txt              # Pinned production dependencies
```

---

## Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/Yasir-Alazmi/enterprise-production-rag.git
cd enterprise-production-rag
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Test Suite (67 Tests)
```bash
python -m pytest tests/ -v
```

### 3. Run Evaluation Harness
```bash
python scripts/evaluate_rag.py
```

### 4. Run Performance Benchmark
```bash
python scripts/benchmark.py
```

### 5. Start Production API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Prometheus Operational Metrics: `http://localhost:8000/api/v1/metrics`
- Health Probe: `http://localhost:8000/api/v1/health`

### 6. Query with Bearer Authentication (cURL)
```bash
# Query with Verified Admin Bearer Token
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-ciso-root" \
  -d '{"query": "What are the restricted incident recovery protocols?", "top_k": 3}'

# Delete a Document (Requires Admin/Manager Bearer Token)
curl -X DELETE http://localhost:8000/api/v1/documents/enterprise_cloud_security_sla \
  -H "Authorization: Bearer token-admin-restricted"

# Run Evaluation Benchmark (Requires Admin/Manager Bearer Token)
curl -X POST http://localhost:8000/api/v1/eval \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-admin-restricted" \
  -d '{"retrieved_ids": ["c1"], "ground_truth_ids": ["c1"], "answer": "Annual leave is 30 days.", "context_chunks": ["Annual vacation entitlement is 30 days."]}'
```

---

## Author & Contact

**Yasir Alazmi**  
Artificial Intelligence Engineer  
- Email: [yasir.alazmi@outlook.sa](mailto:yasir.alazmi@outlook.sa)  
- LinkedIn: [yasir-alazmi-471832436](https://www.linkedin.com/in/yasir-alazmi-471832436)  
- GitHub: [Yasir-Alazmi](https://github.com/Yasir-Alazmi)
