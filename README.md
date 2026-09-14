# Enterprise RAG Reference Architecture

[![CI](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/enterprise-production-rag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

An enterprise reference architecture and engineering blueprint for high-security, role-based hybrid Retrieval-Augmented Generation (RAG). Built for mission-critical enterprise environments requiring zero-trust identity verification, hierarchical RBAC filtering inside retrieval, adversarial injection defense, PII sanitization, dual-mode embedding engines, pluggable LLM generation abstractions, atomic disk persistence, and comprehensive Prometheus telemetry.

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
        [ Protected FastAPI Gateway (/api/v1/query, /ingest, /documents/{id}) ]
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
                  [ Cross-Encoder Re-Ranking ]
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

### 1. Zero-Trust Identity & Cryptographic Access Control (`src/core/security.py`)
- **HMAC-SHA256 JWT Decoding**: Validates cryptographic signatures, token expiration (`exp`), and tenant clearance claims (`sub`, `role`, `clearance`).
- **Protected Administrative Endpoints**: `POST /api/v1/ingest` and `DELETE /api/v1/documents/{id}` strictly require Bearer authorization with $\ge$ `CONFIDENTIAL` clearance. Unauthenticated calls return `HTTP 401 Unauthorized`; insufficient clearance calls return `HTTP 403 Forbidden`.
- **Strict Header Enforcement**: Unrecognized or malformed tokens explicitly return `401 Unauthorized` rather than silently degrading permissions.

### 2. Dual-Mode Embedding Engine Interface (`src/retrieval/embeddings.py`)
- **Interface Contract (`BaseEmbeddingEngine`)**: Decouples the vector store and cache from underlying embedding algorithms.
- **`DeterministicHashingEmbedding` (128-d)**: High-performance, zero-dependency tokenized sign-hash projection into a normalized unit hypersphere. Enables ultra-fast offline CI matrices and reproducible benchmarks without paid API keys or GPU overhead.
- **`SentenceTransformerEmbedding` (384-d)**: Neural semantic embeddings using Sentence Transformers (`all-MiniLM-L6-v2`) with graceful fallback.

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

### Performance Benchmark (`scripts/benchmark.py`)
Measured locally across 100 consecutive requests on Windows (AMD64, Python 3.13):

| Performance Metric | Cold Hybrid Retrieval + Synthesis | Semantic Cache Hit |
| :--- | :--- | :--- |
| **p50 Latency (Median)** | **2.37 ms** | **2.06 ms** |
| **p95 Latency** | **2.86 ms** | **2.44 ms** |
| **p99 Latency** | **4.02 ms** | **2.67 ms** |
| **Mean Execution Time** | **2.44 ms** | **2.08 ms** |
| **Throughput Capacity** | > 400 req/sec | > 500 req/sec |
| **Automated Test Suite** | **60 / 60 Passed (100%)** | 13 Test Suites |

> **Benchmark Scope Note**: Latency covers full Ingestion -> PII Sanitization -> Injection Screening -> Role-Scoped Cache -> Hybrid Retrieval -> Cross-Encoder -> Deterministic Grounded Synthesis. Live third-party external LLM API calls incur separate network roundtrips (~300ms - 1500ms).

### Automated RAG Evaluation Report (`scripts/evaluate_rag.py`)
Evaluated against the Golden Reference Corpus across multi-topic enterprise domains:

| Evaluation Metric | Aggregate Score | Target SLA |
| :--- | :--- | :--- |
| **Mean Context Precision@K** | **0.833** | > 0.80 |
| **Mean Context Recall@K** | **1.000** | > 0.90 |
| **Mean Reciprocal Rank (MRR)** | **0.833** | > 0.80 |
| **Mean NDCG@5** | **0.875** | > 0.85 |
| **Answer Faithfulness** | **0.679** | Grounded Context Coverage |
| **Mean Retrieval Latency** | **0.25 ms** | < 5.00 ms |
| **Mean Generation Latency (Local)** | **0.11 ms** | < 5.00 ms |

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
│   ├── reranker/                 # Cross-encoder joint interaction scorer
│   └── retrieval/                # Dual-mode embeddings, BM25, and hybrid RRF retriever
├── tests/                        # 60 automated tests across 13 test suites (100% passing)
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
source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
pip install -r requirements.txt
```

### 2. Run Test Suite (60 Tests)
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
```

---

## Author & Contact

**Yasir Alazmi**  
Artificial Intelligence Engineer  
- Email: [yasir.alazmi@outlook.sa](mailto:yasir.alazmi@outlook.sa)  
- LinkedIn: [yasir-alazmi-471832436](https://www.linkedin.com/in/yasir-alazmi-471832436)  
- GitHub: [Yasir-Alazmi](https://github.com/Yasir-Alazmi)
