# Enterprise Production RAG Engine

[![CI](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

Production-grade, enterprise-scale hybrid Retrieval-Augmented Generation (RAG) platform. Features multi-source generative answer synthesis, BM25 lexical precision + dense semantic vector search, cross-encoder re-ranking, role-scoped semantic caching, HTTP Bearer Token identity verification, atomic disk persistence, Prometheus operational telemetry, rate limiting, and automated evaluation metrics.

---

## Architecture Overview

```
[ Client Request ] ──► [ Sliding Window Rate Limiter (200 req/min) ]
                               │
                               ▼
        [ Bearer Token Identity & RBAC Clearance Resolver ]
             (token-ciso-root / token-employee-internal)
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
            [ Role-Scoped Bounded Semantic Cache ] ──(Hit)──► Return Cached Answer (<2ms)
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
            [ Multi-Source Generative Synthesis Engine ]
            (Factual Stitching, Strict Refusal, Citations)
                               │
                               ▼
            [ Prometheus Telemetry (/api/v1/metrics) ]
```

---

## Core Production Features

1. **Multi-Source Generative Answer Synthesis (`src/generation/`)**:
   - Rather than returning raw text chunks, the synthesis engine combines facts across multiple retrieved chunks into an executive response.
   - **Hallucination Defense**: If retrieved context relevance is below threshold, outputs strict deterministic refusal: *"Insufficient enterprise documentation was retrieved to answer this query with verifiable certainty."*
   - Appends bracketed section citations `[Document Title - Section]` to every factual claim.

2. **Role-Scoped Semantic Cache (Zero Privilege Escalation)**:
   - High-similarity queries bypass inference entirely, returning responses in `< 2ms`.
   - **Clearance Scoping**: Cache entries are cryptographically bound to `ClassificationLevel`. An employee query will never receive a cached answer generated from an administrator's restricted inquiry.
   - Memory-protected with bounded LRU eviction (default 1,000 entries).

3. **Enterprise Identity & Bearer Token Authentication**:
   - Eliminates client JSON role spoofing via HTTP `Authorization: Bearer <token>` header validation.
   - Resolves authentic caller clearance (`PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `RESTRICTED`).

4. **Atomic Disk Persistence & Document Lifecycle Management**:
   - Automated serialization of vector stores and inverted indexes to `data/storage/` with SHA256 integrity checksums.
   - Restores state on cold boot in `< 50ms`.
   - Provides `DELETE /api/v1/documents/{document_id}` for clean document de-indexing and supports idempotent re-ingestion without chunk duplication.

5. **Enterprise Observability & Rate Limiting**:
   - Prometheus metrics endpoint (`/api/v1/metrics`) exposing request counts, p50/p95/p99 duration histograms, and cache hit/eviction statistics.
   - Sliding-window ASGI rate limiter returning `HTTP 429 Too Many Requests`.

---

## Empirical Benchmark Results

Measured locally using the included reproducible benchmark harness (`scripts/benchmark.py`) across 100 consecutive requests:

| Performance Metric | Cold Hybrid Retrieval + Synthesis | Semantic Cache Hit |
| :--- | :--- | :--- |
| **p50 Latency (Median)** | **2.15 ms** | **1.98 ms** |
| **p95 Latency** | **2.71 ms** | **2.42 ms** |
| **p99 Latency** | **3.76 ms** | **2.51 ms** |
| **Mean Execution Time** | **2.23 ms** | **2.02 ms** |
| **Throughput Capacity** | > 400 req/sec | > 500 req/sec |
| **Test Suite Pass Rate** | **36 / 36 Passed (100%)** | Zero Warnings |

> **Benchmark Scope Note**: Latency covers full Ingestion -> Guardrails -> Role-Scoped Cache -> Hybrid Retrieval -> Cross-Encoder -> In-Memory Generative Synthesis (excluding third-party external cloud LLM API network roundtrips).
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
│   ├── cache/                    # Role-scoped bounded LRU semantic query cache
│   ├── core/                     # Bearer Auth, RBAC security, Prometheus metrics
│   ├── evaluation/               # Context precision, recall, and faithfulness scoring
│   ├── generation/               # Multi-source generative synthesis engine
│   ├── guardrails/               # PII sanitization and adversarial injection screening
│   ├── ingestion/                # Document parsing and recursive token chunker
│   ├── reranker/                 # Cross-encoder joint interaction scorer
│   └── retrieval/                # BM25, dense vector store, and hybrid RRF retriever
├── tests/                        # 36 automated test cases across 10 test suites
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

### 5. Query with Bearer Authentication (cURL)
```bash
# Query with Verified Admin Bearer Token
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-ciso-root" \
  -d '{"query": "What are the restricted incident recovery protocols?", "top_k": 3}'

# Delete a Document from all Indexes
curl -X DELETE http://localhost:8000/api/v1/documents/enterprise_cloud_security_sla \
  -H "Authorization: Bearer token-admin-restricted"
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
