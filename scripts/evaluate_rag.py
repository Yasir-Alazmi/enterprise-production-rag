"""Comprehensive RAG Evaluation Harness.

Evaluates empirical information retrieval and grounded generation metrics:
- Context Precision@K
- Context Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@5)
- Answer Faithfulness
- Answer Relevance
- Stage latency breakdown (Retrieval vs. Generation)

Outputs findings to results/evaluation_report.json.
"""

import json
import platform
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.evaluation.metrics import RAGEvaluator  # noqa: E402

ADMIN_HEADERS = {"Authorization": "Bearer token-admin-restricted"}
CISO_HEADERS = {"Authorization": "Bearer token-ciso-root"}

# Golden Evaluation Corpus
EVAL_CORPUS = [
    {
        "id": "sec_sla_001",
        "title": "Cloud Security SLA",
        "content": (
            "Enterprise cloud infrastructure requires 99.95% monthly uptime. "
            "Data encryption at rest must use AES-256 with customer-managed keys. "
            "Data in transit requires TLS 1.3 across all communication channels. "
            "Incident response for P1 critical events must occur within 15 minutes."
        ),
        "classification": "INTERNAL",
        "metadata": {"section": "SLA Uptime"}
    },
    {
        "id": "hr_leave_002",
        "title": "Corporate Leave Policy",
        "content": (
            "Annual vacation entitlement is 30 calendar days per annum for full-time employees. "
            "Sick leave allows up to 14 consecutive days with certified medical documentation. "
            "Parental leave provides 12 weeks of fully paid leave for eligible primary caregivers."
        ),
        "classification": "INTERNAL",
        "metadata": {"section": "Leave Policy"}
    },
    {
        "id": "crypto_audit_003",
        "title": "CISO Root Key Ceremony",
        "content": (
            "Root cryptographic master keys are generated using an air-gapped FIPS 140-2 Level 4 HSM. "
            "Dual-custody M-of-N key ceremonies require at least three authorized security officers in the Zurich vault. "
            "Key rotation occurs strictly every 365 calendar days."
        ),
        "classification": "RESTRICTED",
        "metadata": {"section": "Key Ceremony"}
    }
]

# Golden Test Evaluation Queries
EVAL_QUERIES = [
    {
        "query": "What is the mandatory monthly uptime requirement for enterprise cloud infrastructure?",
        "ground_truth_ids": ["sec_sla_001"],
        "auth_headers": {"Authorization": "Bearer token-employee-internal"},
        "expected_keywords": ["99.95%", "uptime"]
    },
    {
        "query": "How many days of annual vacation are full-time employees entitled to?",
        "ground_truth_ids": ["hr_leave_002"],
        "auth_headers": {"Authorization": "Bearer token-employee-internal"},
        "expected_keywords": ["30 calendar days", "vacation"]
    },
    {
        "query": "What are the security requirements for the root HSM key ceremony in Zurich?",
        "ground_truth_ids": ["crypto_audit_003"],
        "auth_headers": CISO_HEADERS,
        "expected_keywords": ["FIPS 140-2", "Zurich vault"]
    },
    {
        "query": "What is the required response time for P1 critical security incidents?",
        "ground_truth_ids": ["sec_sla_001"],
        "auth_headers": {"Authorization": "Bearer token-employee-internal"},
        "expected_keywords": ["15 minutes", "P1"]
    }
]


def run_evaluation() -> dict:
    print("=== Enterprise RAG Reference Architecture: Automated Evaluation Harness ===")
    evaluator = RAGEvaluator()

    with TestClient(app) as client:
        # 1. Ingest Golden Corpus
        print("[*] Ingesting Golden Reference Corpus...")
        ingest_res = client.post(
            "/api/v1/ingest",
            json={"documents": EVAL_CORPUS, "persist_to_disk": False},
            headers=ADMIN_HEADERS
        )
        assert ingest_res.status_code == 201, f"Ingest failed: {ingest_res.text}"

        query_eval_results = []
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        ndcg_scores = []
        faithfulness_scores = []
        relevance_scores = []
        retrieval_latencies = []
        generation_latencies = []

        print(f"[*] Executing {len(EVAL_QUERIES)} Benchmark Evaluation Queries...")
        for item in EVAL_QUERIES:
            q = item["query"]
            gt = item["ground_truth_ids"]
            headers = item["auth_headers"]

            t0 = time.perf_counter()
            res = client.post(
                "/api/v1/query",
                json={"query": q, "top_k": 3, "enable_cache": False},
                headers=headers
            )
            total_dur = (time.perf_counter() - t0) * 1000.0
            assert res.status_code == 200, f"Query failed: {res.text}"
            data = res.json()

            retrieved_doc_ids = [c["document_id"] for c in data.get("citations", [])]
            retrieved_snippets = [c["content_snippet"] for c in data.get("citations", [])]
            answer = data.get("answer", "")

            metrics = evaluator.evaluate_query(
                retrieved_ids=retrieved_doc_ids,
                ground_truth_ids=gt,
                answer=answer,
                context_chunks=retrieved_snippets,
                query=q
            )

            ret_ms = data.get("retrieval_latency_ms", 0.0)
            gen_ms = data.get("generation_latency_ms", 0.0)

            precision_scores.append(metrics["context_precision"])
            recall_scores.append(metrics["context_recall"])
            mrr_scores.append(metrics["mrr"])
            ndcg_scores.append(metrics["ndcg_at_5"])
            faithfulness_scores.append(metrics["faithfulness"])
            relevance_scores.append(metrics.get("answer_relevance", 0.0))
            retrieval_latencies.append(ret_ms)
            generation_latencies.append(gen_ms)

            query_eval_results.append({
                "query": q,
                "retrieved_doc_ids": retrieved_doc_ids,
                "ground_truth_ids": gt,
                "answer": answer,
                "metrics": metrics,
                "latency_breakdown_ms": {
                    "total": round(total_dur, 2),
                    "retrieval": ret_ms,
                    "generation": gen_ms
                }
            })

    summary = {
        "metadata": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries_evaluated": len(EVAL_QUERIES)
        },
        "aggregate_metrics": {
            "mean_context_precision": round(sum(precision_scores) / len(precision_scores), 4),
            "mean_context_recall": round(sum(recall_scores) / len(recall_scores), 4),
            "mean_mrr": round(sum(mrr_scores) / len(mrr_scores), 4),
            "mean_ndcg_at_5": round(sum(ndcg_scores) / len(ndcg_scores), 4),
            "mean_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
            "mean_answer_relevance": round(sum(relevance_scores) / len(relevance_scores), 4),
        },
        "latency_breakdown_averages_ms": {
            "retrieval_mean_ms": round(sum(retrieval_latencies) / len(retrieval_latencies), 2),
            "generation_mean_ms": round(sum(generation_latencies) / len(generation_latencies), 2),
        },
        "query_details": query_eval_results
    }

    print("\n" + "=" * 76)
    print(f"{'EVALUATION METRIC':<35} | {'AGGREGATE VALUE':<20} | {'TARGET SLA':<15}")
    print("-" * 76)
    print(f"{'Mean Context Precision@K':<35} | {summary['aggregate_metrics']['mean_context_precision']:<20} | {'> 0.85':<15}")
    print(f"{'Mean Context Recall@K':<35} | {summary['aggregate_metrics']['mean_context_recall']:<20} | {'> 0.90':<15}")
    print(f"{'Mean Reciprocal Rank (MRR)':<35} | {summary['aggregate_metrics']['mean_mrr']:<20} | {'> 0.85':<15}")
    print(f"{'Mean NDCG@5':<35} | {summary['aggregate_metrics']['mean_ndcg_at_5']:<20} | {'> 0.85':<15}")
    print(f"{'Answer Faithfulness':<35} | {summary['aggregate_metrics']['mean_faithfulness']:<20} | {'1.00 (Zero Halluc)':<15}")
    print(f"{'Answer Keyword Relevance':<35} | {summary['aggregate_metrics']['mean_answer_relevance']:<20} | {'> 0.50':<15}")
    print("-" * 76)
    print(f"{'Mean Retrieval Latency':<35} | {summary['latency_breakdown_averages_ms']['retrieval_mean_ms']:<17} ms | {'< 5.00 ms':<15}")
    print(f"{'Mean Generation Latency (Local)':<35} | {summary['latency_breakdown_averages_ms']['generation_mean_ms']:<17} ms | {'< 5.00 ms':<15}")
    print("=" * 76 + "\n")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Detailed evaluation report saved to: {out_file.relative_to(PROJECT_ROOT)}")
    return summary


if __name__ == "__main__":
    run_evaluation()
