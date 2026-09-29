from src.agents.base import SkeletonAgent
from src.contracts.enums import AgentId


class DiagnosisAgent(SkeletonAgent):
    agent_id = AgentId.DIAGNOSIS
    responsibility = "Rank root causes and recommend evidence-backed actions"
