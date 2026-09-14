# Enterprise Production RAG: Operational Runbook

This document details day-2 operations, disaster recovery protocols, observability integration, and maintenance procedures.

---

## 1. Disaster Recovery & Snapshot Persistence
The platform maintains atomic state persistence in `data/storage/`.

### Manual Snapshot Trigger
To force an atomic snapshot of current vector embeddings and BM25 indices:
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [], "persist_to_disk": true}'
```

### Snapshot Restoration & Integrity Verification
Upon container boot, the `lifespan` hook automatically verifies `data/storage/vector_store.json` using SHA256 integrity hashes:
1. If the hash matches, the index is loaded in `< 50ms`.
2. If corrupted, the snapshot is quarantined, an alert is logged, and sample documents are rebuilt automatically.

---

## 2. Role-Based Access Control (RBAC) Matrix

| Classification Tier | Minimum Clearance Role | Allowed Resources |
| :--- | :--- | :--- |
| **PUBLIC (0)** | `anonymous`, `guest` | Public press releases, external FAQ, open policies. |
| **INTERNAL (1)** | `employee`, `contractor` | General employee handbook, internal Slack guidelines. |
| **CONFIDENTIAL (2)** | `manager`, `compliance_officer` | Security SLAs, Disaster Recovery runbooks, audits. |
| **RESTRICTED (3)** | `admin`, `ciso`, `executive` | Board resolutions, cryptographic key schemes, HR salary bands. |

---

## 3. Prometheus Observability & Grafana Integration

The service exposes metrics at `/api/v1/metrics` in standard Prometheus text format.

### Recommended Prometheus Scrape Config
```yaml
scrape_configs:
  - job_name: 'enterprise_rag_engine'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/metrics'
```

### Key Alerting Rules
- **High Latency Alert**: `rag_request_duration_seconds{quantile="0.95"} > 0.100` (p95 exceeding 100ms for 5m).
- **Cache Inefficiency Alert**: `rate(rag_cache_misses_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m])) > 0.80`.

---

## 4. Rate Limiting Configuration
Rate limiting is enforced at the ASGI middleware level using a sliding-window algorithm:
- Default quota: `200 requests/minute` per client IP.
- When exceeded, returns `HTTP 429 Too Many Requests` with header `Retry-After: 60`.
