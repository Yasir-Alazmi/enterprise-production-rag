"""Rate limiting and telemetry middleware."""

import time
from collections import defaultdict
from typing import Dict, List

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.metrics import metrics_collector


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """In-memory sliding window rate limiter protecting endpoints against abuse."""

    def __init__(self, app, max_requests_per_minute: int = 120):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.clients: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Exclude health and metrics from rate limiting
        if request.url.path in ["/api/v1/health", "/api/v1/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        window_start = now - 60.0

        # Purge requests outside sliding window
        self.clients[client_ip] = [ts for ts in self.clients[client_ip] if ts > window_start]

        if len(self.clients[client_ip]) >= self.max_requests:
            return Response(
                content='{"error": "RateLimitExceeded", "message": "Too many requests. Please retry in 60 seconds."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": "60"}
            )

        self.clients[client_ip].append(now)
        return await call_next(request)

class TelemetryMiddleware(BaseHTTPMiddleware):
    """Measures request duration and feeds telemetry into Prometheus metrics collector."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        metrics_collector.record_request(request.url.path, response.status_code, duration)
        return response
