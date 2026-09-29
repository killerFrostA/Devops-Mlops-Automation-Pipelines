from src.agents.base import SkeletonAgent
from src.contracts.enums import AgentId


class SecurityQualityAgent(SkeletonAgent):
    agent_id = AgentId.SECURITY_QUALITY
    responsibility = "Aggregate test, quality, dependency, image and secret scan evidence"
