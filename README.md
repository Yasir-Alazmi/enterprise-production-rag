# Enterprise Production RAG Engine

[![CI](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

Production-grade, enterprise-scale hybrid Retrieval-Augmented Generation (RAG) platform. Features BM25 lexical precision + dense semantic vector search, cross-encoder re-ranking, role-based access control (RBAC), atomic disk persistence, bounded LRU semantic query caching, Prometheus operational telemetry, rate limiting, and automated evaluation metrics.

---

## Architecture Overview

```
[ Client Request ] ──► [ Sliding Window Rate Limiter (200 req/min) ]
                               │
                               ▼
              [ FastAPI Gateway (/api/v1/query) ]
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[ Injection Detector ]                         [ PII Sanitizer ]
(System Breakouts)                             (National ID, Cards, Keys)
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
               [ Bounded LRU Semantic Cache ] ──(Hit)──► Return Cached Answer (<2ms)
                               │ (Miss)
                               ▼
           [ Role-Based Access Control (RBAC Filter) ]
           (PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED)
                               │
                               ▼
             [ Hybrid Retrieval Engine (RRF Fusion) ]
             ├── BM25Okapi Lexical Index
             └── Dense Vector Store (Cosine Similarity)
                               │
                               ▼
                 [ Cross-Encoder Re-Ranking ]
                 (Joint Query-Document Interaction)
                               │
                               ▼
              [ Grounded Answer & Citations Generator ]
                               │
                               ▼
            [ Prometheus Telemetry (/api/v1/metrics) ]
```

---

## Core Production Features

1. **Role-Based Access Control (RBAC) & Multi-Tenancy**:
   - Enforces hierarchical document clearances (`PUBLIC` -> `INTERNAL` -> `CONFIDENTIAL` -> `RESTRICTED`).
   - Query filters isolate unauthorized document chunks before scoring, guaranteeing zero cross-tenant data leakage.

2. **Atomic Disk Persistence & State Durability**:
   - Automated serialization of vector stores and inverted indexes to `data/storage/` with SHA256 payload integrity checksums.
   - Graceful state restoration on cold boot in `< 50ms`.

3. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - Combines lexical precision (BM25 for exact codes, statutes, and acronyms) with dense vector semantics via configurable weight alpha = 0.60.

4. **Cross-Encoder Re-Ranking**:
   - Evaluates joint query-document coverage and term proximity, eliminating semantic false positives where general topic aligns but specific facts are absent.

5. **Bounded LRU Semantic Cache**:
   - High-similarity queries bypass inference entirely, returning responses in `< 2ms`.
   - Memory-protected with an LRU eviction policy to eliminate out-of-memory (OOM) risks.

6. **Enterprise Observability & Rate Limiting**:
   - Prometheus metrics endpoint (`/api/v1/metrics`) exposing request counts, p50/p95/p99 duration histograms, and cache statistics.
   - Sliding-window ASGI rate limiter returning `HTTP 429 Too Many Requests`.

---

## Empirical Benchmark Results

Measured locally using the included reproducible benchmark harness (`scripts/benchmark.py`) across 100 consecutive requests:

| Performance Metric | Cold Hybrid Retrieval | Semantic Cache Hit |
| :--- | :--- | :--- |
| **p50 Latency (Median)** | **2.05 ms** | **1.91 ms** |
| **p95 Latency** | **2.58 ms** | **2.36 ms** |
| **p99 Latency** | **3.44 ms** | **2.61 ms** |
| **Mean Execution Time** | **2.13 ms** | **1.97 ms** |
| **Throughput Capacity** | > 400 req/sec | > 500 req/sec |
| **Test Suite Pass Rate** | **30 / 30 Passed (100%)** | Zero Warnings |

> **Reproducibility**: Run `python scripts/benchmark.py` to regenerate `results/benchmark_report.json` with platform hardware telemetry.

---

## Directory Structure

```
enterprise-production-rag/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix (Python 3.10 & 3.11)
├── configs/config.yaml            # Centralized hyperparameter configuration
├── data/
│   ├── sample_documents/         # Compliance & Security SLA documents
│   └── storage/                  # Persistent vector snapshots with SHA256 verification
├── docs/
│   ├── architecture.md           # Engineering trade-offs and design rationale
│   └── runbook.md                # Day-2 operations, disaster recovery, and alerting rules
├── results/
│   └── benchmark_report.json     # Empirical latency percentiles and system metadata
├── scripts/
│   └── benchmark.py              # Standalone benchmark harness (p50/p95/p99)
├── src/
│   ├── api/                      # FastAPI routes, schemas, and rate-limiting middleware
│   ├── cache/                    # Bounded LRU semantic query cache
│   ├── core/                     # RBAC security, Prometheus metrics, Pydantic settings
│   ├── evaluation/               # Context precision, recall, and faithfulness scoring
│   ├── guardrails/               # PII sanitization and adversarial injection screening
│   ├── ingestion/                # Document parsing and recursive token chunker
│   ├── reranker/                 # Cross-encoder joint interaction scorer
│   └── retrieval/                # BM25, dense vector store, and hybrid RRF retriever
├── tests/                        # 30 automated test cases across 9 test suites
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

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Run Benchmark Harness
```bash
python scripts/benchmark.py
```

### 4. Start Production API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger Interactive Documentation: `http://localhost:8000/docs`
- Prometheus Operational Metrics: `http://localhost:8000/api/v1/metrics`
- Health Probe: `http://localhost:8000/api/v1/health`

### 5. Query with RBAC Clearance (cURL)
```bash
# Query with Employee Clearance
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the monthly SLA uptime?", "user_role": "employee", "top_k": 3}'

# Query with Executive/Admin Clearance
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the restricted incident recovery protocols?", "user_role": "admin", "top_k": 3}'
```

---

## Operational Runbook & Compliance

For details on disaster recovery procedures, Prometheus scrape configurations, and Saudi PDPL compliance controls, refer to [`docs/runbook.md`](docs/runbook.md).

---

## Author & Contact

**Yasir Alazmi**  
Artificial Intelligence Engineer  
- Email: [yasir.alazmi@outlook.sa](mailto:yasir.alazmi@outlook.sa)  
- LinkedIn: [yasir-alazmi-471832436](https://www.linkedin.com/in/yasir-alazmi-471832436)  
- GitHub: [Yasir-Alazmi](https://github.com/Yasir-Alazmi)
