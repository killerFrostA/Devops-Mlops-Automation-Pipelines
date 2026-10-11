"""How to measure one service, so Agent 1 applies to any project without code changes.

Two modes:
- app: the service exposes its own Prometheus metrics (/metrics). Every signal is available:
  traffic, errors, latency (SLO), memory and CPU. Metric and label names are configurable,
  because projects do not all use the same instrumentation library.
- container: the service exposes nothing. cAdvisor measures its Docker container from the
  outside: memory and CPU only (no request counter, so no SLO). Logs work in both modes.

Every name here is interpolated into PromQL, so each one is validated.
"""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

_METRIC = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_CONTAINER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


class Mode(StrEnum):
    APP = "app"
    CONTAINER = "container"


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricNames(_Config):
    """Names used by the service's own metrics. Defaults = prometheus_client conventions."""

    requests: str = "http_requests_total"  # counter of requests
    latency_histogram: str = "http_request_duration_seconds"  # histogram (_bucket, _count)
    memory_bytes: str = "process_resident_memory_bytes"
    cpu_seconds: str = "process_cpu_seconds_total"
    status_label: str = "status"  # HTTP status code label, e.g. "code" in other libraries
    path_label: str = "path"  # route label, e.g. "handler"
    # Requests that are not user traffic (probes, scrapes, fault injection). Empty: keep all.
    excluded_paths: str = "/health.*|/metrics|/chaos.*"

    @field_validator("requests", "latency_histogram", "memory_bytes", "cpu_seconds")
    @classmethod
    def metric_name(cls, value: str) -> str:
        if not _METRIC.fullmatch(value):
            raise ValueError(f"Invalid metric name: {value!r}")
        return value

    @field_validator("status_label", "path_label")
    @classmethod
    def label_name(cls, value: str) -> str:
        if not _LABEL.fullmatch(value):
            raise ValueError(f"Invalid label name: {value!r}")
        return value

    @field_validator("excluded_paths")
    @classmethod
    def path_regex(cls, value: str) -> str:
        if '"' in value or "\\" in value:
            raise ValueError("excluded_paths must not contain quotes or backslashes")
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(f"Invalid excluded_paths regex: {error}") from error
        return value


class ServiceProfile(_Config):
    mode: Mode = Mode.APP
    metrics: MetricNames = MetricNames()
    # container mode: Docker container name as seen by cAdvisor (default: the service_id).
    container: str | None = None

    @field_validator("container")
    @classmethod
    def container_name(cls, value: str | None) -> str | None:
        if value is not None:
            check_container(value)
        return value


def check_container(name: str) -> None:
    if not _CONTAINER.fullmatch(name):
        raise ValueError(f"Invalid container name: {name!r}")
