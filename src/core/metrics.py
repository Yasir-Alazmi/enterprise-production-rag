"""Prometheus metrics and enterprise telemetry collector."""

from collections import Counter
from typing import Any, Dict, List

import numpy as np


class PrometheusMetrics:
    """Collects and formats operational telemetry in standard Prometheus exposition format."""

    def __init__(self):
        self.request_counters: Counter = Counter()
        self.latencies: List[float] = []

    def record_request(self, endpoint: str, status_code: int, duration_seconds: float) -> None:
        """Record an API request event with its execution duration."""
        key = f'{endpoint}:{status_code}'
        self.request_counters[key] += 1
        self.latencies.append(duration_seconds)
        # Cap sliding latency window to 10,000 samples
        if len(self.latencies) > 10000:
            self.latencies.pop(0)

    def get_percentiles(self) -> Dict[str, float]:
        """Compute empirical latency percentiles (p50, p95, p99)."""
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        arr = np.array(self.latencies)
        return {
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "p99": round(float(np.percentile(arr, 99)), 6),
            "mean": round(float(np.mean(arr)), 6)
        }

    def export_text(self, cache_stats: Dict[str, Any], indexed_chunks: int) -> str:
        """Export metrics formatted for Prometheus scraping."""
        lines = [
            "# HELP rag_http_requests_total Total number of HTTP requests processed.",
            "# TYPE rag_http_requests_total counter"
        ]
        for key, count in self.request_counters.items():
            endpoint, status = key.split(":")
            lines.append(f'rag_http_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}')

        percentiles = self.get_percentiles()
        lines.extend([
            "# HELP rag_request_duration_seconds HTTP request execution duration percentiles.",
            "# TYPE rag_request_duration_seconds summary",
            f'rag_request_duration_seconds{{quantile="0.5"}} {percentiles["p50"]}',
            f'rag_request_duration_seconds{{quantile="0.95"}} {percentiles["p95"]}',
            f'rag_request_duration_seconds{{quantile="0.99"}} {percentiles["p99"]}',
            f'rag_request_duration_seconds_sum {round(sum(self.latencies), 4)}',
            f'rag_request_duration_seconds_count {len(self.latencies)}'
        ])

        lines.extend([
            "# HELP rag_cache_hits_total Total semantic query cache hits.",
            "# TYPE rag_cache_hits_total counter",
            f'rag_cache_hits_total {cache_stats.get("hits", 0)}',
            "# HELP rag_cache_misses_total Total semantic query cache misses.",
            "# TYPE rag_cache_misses_total counter",
            f'rag_cache_misses_total {cache_stats.get("misses", 0)}',
            "# HELP rag_cache_evictions_total Total LRU evictions from query cache.",
            "# TYPE rag_cache_evictions_total counter",
            f'rag_cache_evictions_total {cache_stats.get("evictions", 0)}',
            "# HELP rag_indexed_chunks_total Total active text chunks indexed.",
            "# TYPE rag_indexed_chunks_total gauge",
            f'rag_indexed_chunks_total {indexed_chunks}'
        ])

        return "\n".join(lines) + "\n"

metrics_collector = PrometheusMetrics()
