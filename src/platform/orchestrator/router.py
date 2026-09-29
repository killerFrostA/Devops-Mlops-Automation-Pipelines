from src.contracts.enums import AgentId
from src.contracts.events import EventEnvelope

# Only safe initial analysis branches are wired in the skeleton.
ANALYSIS_ROUTES: dict[str, tuple[AgentId, ...]] = {
    "monitoring.anomaly.detected": (AgentId.DIAGNOSIS,),
    "diagnosis.completed": (AgentId.SECURITY_QUALITY, AgentId.RESOURCE_OPTIMIZATION),
}


def route(event: EventEnvelope) -> tuple[AgentId, ...]:
    """Return planning targets. This function neither dispatches nor executes actions."""
    return ANALYSIS_ROUTES.get(event.topic, ())
