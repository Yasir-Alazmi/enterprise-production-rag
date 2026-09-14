"""Reproducible performance benchmark harness measuring latency percentiles and throughput."""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402


def run_benchmark(num_iterations: int = 100) -> dict:
    print(f"=== Enterprise Production RAG: Performance Benchmark ({num_iterations} iterations) ===")

    results = {
        "metadata": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "cold_queries": {},
        "cached_queries": {}
    }

    with TestClient(app) as client:
        # Pre-warm index
        client.get("/api/v1/health")

        # 1. Benchmark Cold Hybrid Queries
        cold_latencies = []
        for i in range(num_iterations):
            # Unique variation to ensure cache misses
            q = f"What is the encryption standard for data at rest under policy variation {i}?"
            start = time.perf_counter()
            res = client.post("/api/v1/query", json={"query": q, "top_k": 3, "enable_cache": False})
            dur = (time.perf_counter() - start) * 1000.0
            if res.status_code == 200:
                cold_latencies.append(dur)

        cold_arr = np.array(cold_latencies)
        results["cold_queries"] = {
            "iterations": len(cold_latencies),
            "p50_ms": round(float(np.percentile(cold_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(cold_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(cold_arr, 99)), 2),
            "mean_ms": round(float(np.mean(cold_arr)), 2),
            "min_ms": round(float(np.min(cold_arr)), 2),
            "max_ms": round(float(np.max(cold_arr)), 2),
            "std_ms": round(float(np.std(cold_arr)), 2)
        }

        # 2. Benchmark Semantic Cache Hit Queries
        cached_latencies = []
        # Pre-seed query
        client.post("/api/v1/query", json={"query": "What is the monthly SLA uptime?", "enable_cache": True})

        for _ in range(num_iterations):
            start = time.perf_counter()
            res = client.post("/api/v1/query", json={"query": "What is the monthly SLA uptime?", "enable_cache": True})
            dur = (time.perf_counter() - start) * 1000.0
            if res.status_code == 200 and res.json().get("cached"):
                cached_latencies.append(dur)

        cached_arr = np.array(cached_latencies)
        results["cached_queries"] = {
            "iterations": len(cached_latencies),
            "p50_ms": round(float(np.percentile(cached_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(cached_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(cached_arr, 99)), 2),
            "mean_ms": round(float(np.mean(cached_arr)), 2),
            "min_ms": round(float(np.min(cached_arr)), 2),
            "max_ms": round(float(np.max(cached_arr)), 2),
            "std_ms": round(float(np.std(cached_arr)), 2)
        }

    # Print Summary Table
    print("\n----------------------------------------------------------------------")
    print("| Metric                     | Cold Hybrid Search  | Semantic Cache Hit |")
    print("----------------------------------------------------------------------")
    print(f"| p50 Latency (Median)       | {results['cold_queries']['p50_ms']:>16} ms | {results['cached_queries']['p50_ms']:>15} ms |")
    print(f"| p95 Latency                | {results['cold_queries']['p95_ms']:>16} ms | {results['cached_queries']['p95_ms']:>15} ms |")
    print(f"| p99 Latency                | {results['cold_queries']['p99_ms']:>16} ms | {results['cached_queries']['p99_ms']:>15} ms |")
    print(f"| Mean Latency               | {results['cold_queries']['mean_ms']:>16} ms | {results['cached_queries']['mean_ms']:>15} ms |")
    print(f"| Min / Max                  | {results['cold_queries']['min_ms']:>7} / {results['cold_queries']['max_ms']:<6} ms | {results['cached_queries']['min_ms']:>6} / {results['cached_queries']['max_ms']:<5} ms |")
    print("----------------------------------------------------------------------\n")

    # Export to JSON
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results exported to: {out_file.relative_to(Path.cwd()) if out_file.is_relative_to(Path.cwd()) else out_file}")
    return results

if __name__ == "__main__":
    run_benchmark(100)
