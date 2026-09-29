from src.agents.base import SkeletonAgent
from src.contracts.enums import AgentId


class ResourceOptimizationAgent(SkeletonAgent):
    agent_id = AgentId.RESOURCE_OPTIMIZATION
    responsibility = "Forecast demand and recommend resource changes within policy bounds"
