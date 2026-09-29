"""One authoritative mapping for supported v1 events."""

from src.contracts.actions import ActionRequest
from src.contracts.base import Contract
from src.contracts.payloads import (
    ActionCompleted,
    AnomalyDetected,
    ApprovalRequested,
    ApprovalResolved,
    DiagnosisCompleted,
    ModelHealthAssessed,
    RecoveryVerified,
    ReleaseAssessment,
    ResourceRecommendation,
    WorkflowContinuation,
)

TOPICS: dict[str, tuple[str, type[Contract]]] = {
    "monitoring.anomaly.detected": ("ANOMALY_DETECTED", AnomalyDetected),
    "monitoring.recovery.verified": ("RECOVERY_VERIFIED", RecoveryVerified),
    "diagnosis.completed": ("DIAGNOSIS_COMPLETED", DiagnosisCompleted),
    "quality.release.assessed": ("RELEASE_ASSESSMENT", ReleaseAssessment),
    "deployment.action.requested": ("ACTION_REQUESTED", ActionRequest),
    "deployment.action.completed": ("ACTION_COMPLETED", ActionCompleted),
    "resource.recommendation.created": ("RESOURCE_RECOMMENDATION", ResourceRecommendation),
    "mlops.model.health.assessed": ("MODEL_HEALTH_ASSESSED", ModelHealthAssessed),
    "approval.requested": ("APPROVAL_REQUESTED", ApprovalRequested),
    "approval.resolved": ("APPROVAL_RESOLVED", ApprovalResolved),
    "workflow.continuation": ("WORKFLOW_CONTINUATION", WorkflowContinuation),
}
