from enum import StrEnum


class AgentId(StrEnum):
    MONITORING = "monitoring-agent"
    DIAGNOSIS = "diagnosis-agent"
    SECURITY_QUALITY = "security-quality-agent"
    DEPLOYMENT_RECOVERY = "deployment-recovery-agent"
    RESOURCE_OPTIMIZATION = "resource-optimization-agent"
    MLOPS_LIFECYCLE = "mlops-lifecycle-agent"


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class Priority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyDecision(StrEnum):
    AUTO = "AUTO"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    BLOCK = "BLOCK"


class ActionType(StrEnum):
    DEPLOY = "DEPLOY"
    ROLLING_RESTART = "ROLLING_RESTART"
    ROLLBACK = "ROLLBACK"
    CANARY = "CANARY"
    BLUE_GREEN_SWITCH = "BLUE_GREEN_SWITCH"
    TRAFFIC_SHIFT = "TRAFFIC_SHIFT"
    SCALE_REPLICAS = "SCALE_REPLICAS"
    PROMOTE_MODEL = "PROMOTE_MODEL"


class ResultStatus(StrEnum):
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
