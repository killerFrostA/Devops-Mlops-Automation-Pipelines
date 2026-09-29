from src.agents.base import SkeletonAgent
from src.contracts.enums import AgentId


class DeploymentRecoveryAgent(SkeletonAgent):
    agent_id = AgentId.DEPLOYMENT_RECOVERY
    responsibility = "Validate authorization and execute exact idempotent deployment actions"
