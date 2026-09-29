from src.agents.base import SkeletonAgent
from src.contracts.enums import AgentId


class MonitoringAgent(SkeletonAgent):
    agent_id = AgentId.MONITORING
    responsibility = "Collect telemetry, detect anomalies and verify recovery"
