"""Prometheus metrics and enterprise telemetry collector with stage duration tracking."""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class PrometheusMetrics:
    """Collects and formats operational telemetry in standard Prometheus exposition format."""

    def __init__(self):
        self.request_counters: Counter = Counter()
        self.latencies: List[float] = []
        self.stage_latencies: Dict[str, List[float]] = defaultdict(list)

    def record_request(self, endpoint: str, status_code: int, duration_seconds: float) -> None:
        """Record an API request event with its execution duration."""
        key = f'{endpoint}:{status_code}'
        self.request_counters[key] += 1
        self.latencies.append(duration_seconds)
        if len(self.latencies) > 10000:
            self.latencies.pop(0)

    def record_stage(self, stage: str, duration_seconds: float) -> None:
        """Record execution duration for specific pipeline stages (e.g., retrieval, generation)."""
        stage_list = self.stage_latencies[stage]
        stage_list.append(duration_seconds)
        if len(stage_list) > 10000:
            stage_list.pop(0)

    def get_percentiles(self, latencies: Optional[List[float]] = None) -> Dict[str, float]:
        """Compute empirical latency percentiles (p50, p95, p99)."""
        data = self.latencies if latencies is None else latencies
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        arr = np.array(data)
        return {
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "p99": round(float(np.percentile(arr, 99)), 6),
            "mean": round(float(np.mean(arr)), 6)
        }

    def export_text(self, cache_stats: Dict[str, Any], indexed_chunks: int) -> str:
        """Export metrics formatted for Prometheus scraping with stage breakdowns."""
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

        # Stage durations (Retrieval vs Generation)
        lines.extend([
            "# HELP rag_stage_duration_seconds Latency broken down by RAG stage (retrieval vs generation).",
            "# TYPE rag_stage_duration_seconds summary"
        ])
        for stage_name, stage_list in self.stage_latencies.items():
            st_p = self.get_percentiles(stage_list)
            lines.extend([
                f'rag_stage_duration_seconds{{stage="{stage_name}",quantile="0.5"}} {st_p["p50"]}',
                f'rag_stage_duration_seconds{{stage="{stage_name}",quantile="0.95"}} {st_p["p95"]}',
                f'rag_stage_duration_seconds{{stage="{stage_name}",quantile="0.99"}} {st_p["p99"]}',
                f'rag_stage_duration_seconds_sum{{stage="{stage_name}"}} {round(sum(stage_list), 4)}',
                f'rag_stage_duration_seconds_count{{stage="{stage_name}"}} {len(stage_list)}'
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
