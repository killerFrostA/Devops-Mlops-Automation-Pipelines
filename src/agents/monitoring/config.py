"""Agent 1 configuration: watched services, cycle timing and detector policies.

Loaded from YAML (configs/monitoring.yaml) and validated, so thresholds change without code
changes. Every section has defaults; an empty file gives a working configuration.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.monitoring.collectors.profile import Mode, ServiceProfile
from src.agents.monitoring.decision import DecisionPolicy
from src.agents.monitoring.detectors.availability import AvailabilityPolicy
from src.agents.monitoring.detectors.forecast import ForecastPolicy
from src.agents.monitoring.detectors.logs import LogPolicy
from src.agents.monitoring.detectors.oom import OomPolicy
from src.agents.monitoring.detectors.slo import SLOPolicy
from src.agents.monitoring.detectors.statistical import StatisticalPolicy
from src.agents.monitoring.events import EventsPolicy
from src.agents.monitoring.memory import MemoryPolicy
from src.agents.monitoring.recovery import RecoveryPolicy


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WatchedService(ServiceProfile):
    """A service to watch, and how to measure it (mode, metric names, container name)."""

    service_id: str
    namespace: str = "local"
    version: str = "unknown"  # deployed version, reported in the events (ServiceIdentity)


class MonitoringConfig(_Config):
    # One full cycle (perceive -> detect -> group -> report) every `interval_seconds`.
    interval_seconds: float = Field(default=60, ge=10, le=600)
    # Upper bound for one service's detectors in a cycle; a slow source cannot stall the loop.
    service_timeout_seconds: float = Field(default=30, gt=0)
    incident_cooldown_seconds: float = Field(default=900, gt=0)
    # Silence after which a returning symptom opens a new incident (and a new alert).
    incident_reopen_after_seconds: float = Field(default=300, gt=0)
    # Open incidents kept on disk (SQLite) to survive a restart; None keeps them in memory.
    incident_store_path: str | None = None
    logs_enabled: bool = True
    services: tuple[WatchedService, ...] = (WatchedService(service_id="payment-service"),)
    availability: AvailabilityPolicy = AvailabilityPolicy()
    oom: OomPolicy = OomPolicy()
    slo: SLOPolicy = SLOPolicy()
    statistical: StatisticalPolicy = StatisticalPolicy()
    forecast: ForecastPolicy = ForecastPolicy()
    logs: LogPolicy = LogPolicy()
    decision: DecisionPolicy = DecisionPolicy()
    memory: MemoryPolicy = MemoryPolicy()
    events: EventsPolicy = EventsPolicy()
    recovery: RecoveryPolicy = RecoveryPolicy()

    def containers(self) -> dict[str, str]:
        """service_id -> Docker container name, for services watched in container mode."""
        return {
            s.service_id: s.container or s.service_id
            for s in self.services
            if s.mode is Mode.CONTAINER
        }

    @model_validator(mode="after")
    def coherent(self) -> "MonitoringConfig":
        if not self.services:
            raise ValueError("At least one service must be watched")
        keys = [(s.service_id, s.namespace) for s in self.services]
        if len(set(keys)) != len(keys):
            raise ValueError("Each (service_id, namespace) must be listed once")
        if self.incident_reopen_after_seconds > self.incident_cooldown_seconds:
            raise ValueError("incident_reopen_after_seconds must be <= incident_cooldown_seconds")
        if self.service_timeout_seconds >= self.interval_seconds:
            raise ValueError("service_timeout_seconds must be shorter than interval_seconds")
        return self


def load_config(path: Path) -> MonitoringConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return MonitoringConfig.model_validate(data)
