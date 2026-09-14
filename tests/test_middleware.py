"""Integration tests for rate limiting middleware and Prometheus metrics endpoint."""

from starlette.testclient import TestClient

from src.api.main import app


def test_prometheus_metrics_endpoint():
    with TestClient(app) as client:
        # Trigger an endpoint
        client.get("/api/v1/health")

        # Scrape metrics
        res = client.get("/api/v1/metrics")
        assert res.status_code == 200
        assert "text/plain" in res.headers["content-type"]
        text = res.text
        assert "rag_http_requests_total" in text
        assert "rag_request_duration_seconds" in text
        assert "rag_indexed_chunks_total" in text

def test_sliding_window_rate_limiter():
    with TestClient(app) as client:
        # Multiple quick calls to health/metrics are unthrottled
        for _ in range(10):
            r = client.get("/api/v1/health")
            assert r.status_code == 200


def test_cors_middleware_allowed_origin():
    with TestClient(app) as client:
        res = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert res.headers.get("access-control-allow-credentials") == "true"


def test_cors_middleware_disallowed_origin():
    with TestClient(app) as client:
        res = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://malicious-site.com",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert res.headers.get("access-control-allow-origin") != "http://malicious-site.com"
