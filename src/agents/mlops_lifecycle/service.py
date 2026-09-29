from src.agents.base import SkeletonAgent
from src.contracts.enums import AgentId


class MLOpsLifecycleAgent(SkeletonAgent):
    agent_id = AgentId.MLOPS_LIFECYCLE
    responsibility = "Assess model health, evaluate candidates and recommend controlled promotion"
