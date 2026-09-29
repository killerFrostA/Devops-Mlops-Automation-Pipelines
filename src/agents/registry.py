from src.agents.base import SkeletonAgent
from src.agents.deployment_recovery.service import DeploymentRecoveryAgent
from src.agents.diagnosis.service import DiagnosisAgent
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.agents.monitoring.service import MonitoringAgent
from src.agents.resource_optimization.service import ResourceOptimizationAgent
from src.agents.security_quality.service import SecurityQualityAgent
from src.contracts.enums import AgentId

AGENTS: dict[AgentId, type[SkeletonAgent]] = {
    AgentId.MONITORING: MonitoringAgent,
    AgentId.DIAGNOSIS: DiagnosisAgent,
    AgentId.SECURITY_QUALITY: SecurityQualityAgent,
    AgentId.DEPLOYMENT_RECOVERY: DeploymentRecoveryAgent,
    AgentId.RESOURCE_OPTIMIZATION: ResourceOptimizationAgent,
    AgentId.MLOPS_LIFECYCLE: MLOpsLifecycleAgent,
}
