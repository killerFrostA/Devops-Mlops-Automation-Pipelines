"""Typed, environment-driven configuration. Secrets stay out of logs."""

from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIDS_", env_file=".env", extra="ignore")

    environment: Literal["local", "test", "staging", "production"] = "local"
    runtime_mode: Literal["skeleton"] = "skeleton"
    service_name: str = "platform-api"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    kafka_bootstrap_servers: str = "localhost:9092"
    postgres_dsn: SecretStr = SecretStr(
        "postgresql://pids:local-development-only@localhost:5432/pids"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    qdrant_url: str = "http://localhost:6333"
    object_storage_endpoint: str = "http://localhost:9000"
    mlflow_tracking_uri: str = "http://localhost:5000"
    prometheus_url: str = "http://localhost:9090"
    loki_url: str = "http://localhost:3100"
    # Optional: without a key, unknown log patterns simply stay UNKNOWN.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    context_grpc_target: str = "localhost:50051"
    hitl_grpc_target: str = "localhost:50052"

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def blank_key_is_absent(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def restrict_skeleton(self) -> "Settings":
        if self.environment not in {"local", "test"}:
            raise ValueError("The skeleton runtime supports local/test environments only")
        return self
