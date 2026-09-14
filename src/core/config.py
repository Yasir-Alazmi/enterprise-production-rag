"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field(default="enterprise-production-rag")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Ingestion
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # Retrieval
    hybrid_alpha: float = Field(default=0.60)
    sparse_k1: float = Field(default=1.5)
    sparse_b: float = Field(default=0.75)
    top_k: int = Field(default=5)
    rerank_top_k: int = Field(default=3)

    # Semantic Cache
    cache_similarity_threshold: float = Field(default=0.92)
    cache_ttl_seconds: int = Field(default=3600)

    # Security & Guardrails
    enable_pii_masking: bool = Field(default=True)
    enable_injection_detection: bool = Field(default=True)
    jwt_secret_key: str = Field(default="enterprise-rag-default-insecure-secret-key-change-in-prod")
    jwt_algorithm: str = Field(default="HS256")
    api_tokens_json: Optional[str] = Field(default=None)

    # Embedding Provider Configuration
    embedding_provider: str = Field(default="deterministic")
    embedding_dimension: int = Field(default=128)

    # Generation Engine Configuration
    generation_provider: str = Field(default="deterministic")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    ollama_base_url: str = Field(default="http://localhost:11434")

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load configuration from a YAML file with environment fallback."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            flat: Dict[str, Any] = {}
            if "app" in raw:
                flat["app_name"] = raw["app"].get("name", "enterprise-production-rag")
                flat["app_env"] = raw["app"].get("environment", "development")
                flat["log_level"] = raw["app"].get("log_level", "INFO")
            if "ingestion" in raw:
                flat["chunk_size"] = raw["ingestion"].get("chunk_size", 512)
                flat["chunk_overlap"] = raw["ingestion"].get("chunk_overlap", 64)
            if "retrieval" in raw:
                flat["hybrid_alpha"] = raw["retrieval"].get("hybrid_alpha", 0.60)
                flat["sparse_k1"] = raw["retrieval"].get("sparse_k1", 1.5)
                flat["sparse_b"] = raw["retrieval"].get("sparse_b", 0.75)
                flat["top_k"] = raw["retrieval"].get("final_top_k", 5)
                flat["rerank_top_k"] = raw["retrieval"].get("rerank_top_k", 3)
            if "guardrails" in raw:
                flat["enable_pii_masking"] = raw["guardrails"].get("enable_pii_sanitization", True)
                flat["enable_injection_detection"] = raw["guardrails"].get("enable_injection_detection", True)
            if "cache" in raw:
                flat["cache_similarity_threshold"] = raw["cache"].get("similarity_threshold", 0.92)
                flat["cache_ttl_seconds"] = raw["cache"].get("ttl_seconds", 3600)

            return cls(**flat)
        return cls()

    def validate_production_security(self) -> None:
        """Enforce strict fail-fast validation in production environments."""
        if self.app_env.lower() in ["production", "prod"]:
            if "default-insecure-secret" in self.jwt_secret_key:
                raise ValueError(
                    "FATAL SECURITY VIOLATION: Cannot start application in production with default insecure "
                    "jwt_secret_key. Configure a cryptographically secure key via the JWT_SECRET_KEY environment variable."
                )

settings = Settings()
