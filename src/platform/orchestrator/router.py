"""Deterministic analysis routing. Action dispatch is handled by separate gates."""

from src.contracts.enums import AgentId
from src.contracts.events import EventEnvelope
from src.contracts.payloads import AnomalyDetected

ANALYSIS_ROUTES: dict[str, tuple[AgentId, ...]] = {
    "monitoring.anomaly.detected": (AgentId.DIAGNOSIS,),
    "diagnosis.completed": (AgentId.SECURITY_QUALITY, AgentId.RESOURCE_OPTIMIZATION),
}


def route(event: EventEnvelope) -> tuple[AgentId, ...]:
    """Return only analysis targets; never route directly to an action owner."""
    targets = ANALYSIS_ROUTES.get(event.topic, ())
    if event.topic == "monitoring.anomaly.detected":
        anomaly = AnomalyDetected.model_validate(event.payload)
        signals = anomaly.signals
        model_id = signals.get("model_id")
        model_version = signals.get("model_version")
        if (
            isinstance(model_id, str)
            and model_id.strip()
            and isinstance(model_version, str)
            and model_version.strip()
        ):
            return (*targets, AgentId.MLOPS_LIFECYCLE)
    return targets
