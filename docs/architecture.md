# Enterprise RAG Reference Architecture: Technical Specifications & Scaling Blueprints

## 1. System Motivation & Architectural Philosophy
Standard consumer LLM integrations and simplistic vector search pipelines face severe operational failure modes in enterprise production:
1. **Dense False Positives**: Vector proximity alone fails to distinguish between topical relevance and factual answer presence.
2. **Vocabulary Mismatch**: Dense embeddings often miss exact part numbers, cryptographic key IDs, error codes, and legal statute numbers.
3. **Privilege Escalation via Caching**: A semantic cache that stores queries without tenant and clearance scoping risks leaking confidential responses to unprivileged users.
4. **Non-Deterministic Hallucinations & High Cost**: Live external LLM calls incur token costs, network roundtrips, and hallucinations when context is insufficient.
5. **Regulatory Non-Compliance**: Lack of pre-retrieval PII sanitization exposes sensitive personal data, violating regulations such as the Saudi Personal Data Protection Law (PDPL) and GDPR.

---

## 2. Multi-Stage Pipeline Architecture

```
+-------------------------------------------------------------------------+
|                              Client Layer                               |
|        REST API / Microservice / Enterprise Dashboard / Mobile          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  Zero-Trust Identity & Token Verification               |
|  - HMAC-SHA256 JWT Token Validation (claims: sub, role, clearance, exp) |
|  - Pre-shared API Bearer Token Registry with Clearance Scoping          |
|  - Strictly rejects invalid/forged credentials with HTTP 401             |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                        Enterprise Guardrails                            |
|  - Injection Detector (AST boundary heuristics, system override blocks) |
|  - PII Sanitizer (National IDs, credit cards, emails, private keys)     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                     Role-Scoped Semantic Query Cache                    |
|  - Normalized Cosine Similarity check against indexed query vectors     |
|  - Privilege Isolation: user_clearance >= cache_entry.clearance_level   |
|  - Bounded LRU eviction policy (1,000 entries)                          |
+------------------------------------+------------------------------------+
                                     | (Cache Miss)
                                     v
+-------------------------------------------------------------------------+
|                      Hybrid Retrieval Engine (RRF)                      |
|  +--------------------------------+  +--------------------------------+ |
|  |     Sparse Search (BM25)       |  |     Dense Vector Retrieval     | |
|  |  Lexical token match (k1, b)   |  |   Pluggable Embedding Engine   | |
|  +--------------------------------+  +--------------------------------+ |
|                                   |  |                                  |
|                                   v  v                                  |
|         Reciprocal Rank Fusion (RRF): score = sum(alpha / (60 + rank))  |
|         Integrated RBAC filter: document_classification <= clearance    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      Cross-Encoder Re-Ranking                           |
|  - Joint Query-Document Coverage & Positional Proximity Scoring         |
|  - Re-sorts top candidates to eliminate semantic false positives       |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               Pluggable Answer Generation Abstraction                   |
|  - BaseAnswerGenerator Protocol:                                        |
|    * DeterministicGroundedGenerator: Extractive factual synthesis,     |
|      bracketed citations [Doc - Sec], deterministic negative refusal.   |
|    * LLMAnswerGenerator: OpenAI / Ollama live answering via HTTPX       |
|      with strict anti-hallucination grounding system prompt.            |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 Prometheus Telemetry & Stage Metrics                    |
|  - rag_stage_duration_seconds{stage="retrieval"}                        |
|  - rag_stage_duration_seconds{stage="generation"}                       |
|  - rag_cache_hits_total, rag_cache_misses_total, rag_cache_evictions   |
+-------------------------------------------------------------------------+
```

---

## 3. Engineering Decisions & Design Rationale

| Architectural Decision | Implementation Choice | Alternative Considered | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Dual-Mode Embedding** | `BaseEmbeddingEngine` (`Deterministic` + `Neural`) | Hardcoded Torch / Cloud API | Provides zero-cost, lightning-fast (<0.3ms) reproducible test matrices for CI while supporting neural models in production. |
| **Answer Generation** | `BaseAnswerGenerator` Protocol | Hardcoded LLM API | Isolates retrieval quality, RBAC, and caching from external LLM nondeterminism, latency spikes, and network downtime. |
| **Query-Time RBAC** | Retrieval Filter (`can_access`) | Post-Retrieval Pruning | Enforcing security during retrieval prevents unprivileged chunks from occupying Top-$K$ candidate slots. |
| **Cache Security** | Role-Scoped Clearance Tags | Unscoped Query Hashing | Prevents privilege escalation where an unprivileged employee retrieves a cached answer originating from an executive query. |
| **Identity Verification** | HMAC-SHA256 JWT + Bearer Header | Request Body JSON `user_role` | Completely eliminates JSON role spoofing; identity is derived cryptographically from verified headers. |
| **Persistence Integrity** | Atomic Write + SHA256 Checksum | In-Place File Overwrite | Protects against index corruption during sudden power termination or container eviction. |

---

## 4. Enterprise Production Scaling Blueprint

For production environments handling millions of documents and concurrent users, the reference implementations map directly to horizontal cloud primitives:

### A. Vector Storage Migration: PostgreSQL + `pgvector`
Replace `DenseVectorStore` with PostgreSQL `pgvector`:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    chunk_id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    classification VARCHAR(16) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384)
);

CREATE INDEX idx_chunks_embedding_hnsw 
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Query with RBAC filtering enforced at SQL engine level:
SELECT chunk_id, document_id, content, 
       1 - (embedding <=> :query_vector) AS cosine_similarity
FROM document_chunks
WHERE classification IN ('PUBLIC', 'INTERNAL') -- based on caller clearance
ORDER BY embedding <=> :query_vector
LIMIT 10;
```

### B. Distributed Semantic Caching: Redis Cluster
Replace `SemanticCache` with RedisVL (RediSearch vector indexing):
- Cluster mode provides horizontal cache partitioning across nodes.
- Clearance levels stored as tag fields (`@clearance:{PUBLIC|INTERNAL|CONFIDENTIAL}`).
- TTL managed natively by Redis memory eviction policies.

### C. Enterprise Identity: OIDC / Keycloak / Azure Entra ID
The JWT validation logic in `src/core/security.py` conforms directly to standard OIDC discovery and JWKS validation:
```python
# Production JWKS integration pattern:
from jwt import PyJWKClient

jwks_client = PyJWKClient("https://auth.company.com/realms/enterprise/protocol/openid-connect/certs")
signing_key = jwks_client.get_signing_key_from_jwt(token)
data = jwt.decode(token, signing_key.key, algorithms=["RS256"], audience="enterprise-rag")
```

### D. Distributed Observability: OpenTelemetry
Pipeline spans are structured to emit OpenTelemetry tracing:
- `trace.span("rag.request")`
  - `trace.span("rag.guardrails.pii")`
  - `trace.span("rag.cache.lookup")`
  - `trace.span("rag.retrieval.hybrid")`
  - `trace.span("rag.reranker")`
  - `trace.span("rag.generation.synthesis")`
