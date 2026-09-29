"""Initial domain interfaces. Detailed detector/scan reports live in immutable evidence."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from src.contracts.actions import ActionIntent, ActionRequest
from src.contracts.base import Contract
from src.contracts.enums import PolicyDecision

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class AnomalyDetected(Contract):
    anomaly_id: str
    detector: str
    anomaly_score: Score
    signals: dict[str, JsonValue]
    verification_required_after_action: bool = True


class RecoveryVerified(Contract):
    verification_id: str
    action_id: str
    window: str
    status: Literal["RECOVERED", "NOT_RECOVERED", "INCONCLUSIVE"]
    checks: dict[str, Literal["PASS", "FAIL", "UNKNOWN"]]


class Hypothesis(Contract):
    rank: Annotated[int, Field(ge=1)]
    code: str
    probability: Score
    evidence_refs: tuple[str, ...]


class DiagnosisCompleted(Contract):
    diagnosis_id: str
    hypotheses: tuple[Hypothesis, ...]
    primary_root_cause: str
    recommended_actions: tuple[ActionIntent, ...]
    rationale_summary: str
    limitations: tuple[str, ...] = ()
    requires_additional_evidence: bool = False


class ReleaseAssessment(Contract):
    assessment_id: str
    commit: str
    artifact_digest: str
    decision: Literal["ALLOW", "ALLOW_WITH_WARNING", "BLOCK"]
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...]
    policy_version: str


class ActionCompleted(Contract):
    action_id: str
    action_version: Annotated[int, Field(ge=1)]
    status: Literal["SUCCESS", "FAILED", "BLOCKED"]
    pre_state: dict[str, JsonValue]
    post_state: dict[str, JsonValue]
    execution_evidence_refs: tuple[str, ...]
    verification_requested: bool = True


class ResourceRecommendation(Contract):
    recommendation_id: str
    candidate_action: ActionIntent
    confidence: Score
    observation_window: str
    rollback_condition: str
    constraints: tuple[str, ...]


class ModelHealthAssessed(Contract):
    assessment_id: str
    model_name: str
    model_version: str
    drift_detected: bool
    drift_score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    decision: Literal["HEALTHY", "COLLECT_LABELS", "RETRAIN_AND_EVALUATE"]
    evidence_refs: tuple[str, ...]
    dataset_ref: str | None = None


class ApprovalRequested(Contract):
    approval_id: str
    intent: ActionIntent
    action_fingerprint: str
    reason: str
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def bind_exact_intent(self) -> "ApprovalRequested":
        if self.action_fingerprint != self.intent.fingerprint():
            raise ValueError("Approval fingerprint must match the exact intent")
        return self


class ApprovalResolved(Contract):
    approval_id: str
    action_id: str
    action_version: Annotated[int, Field(ge=1)]
    decision: Literal["APPROVED", "REJECTED", "EXPIRED"]
    # Production actor identity must come from the authenticated principal.
    approver_id: str | None = None
    comment: str


class WorkflowContinuation(Contract):
    incident_id: str
    next_step: Literal["EXECUTE_ACTION", "REPLAN", "CLOSE"]
    action_request: ActionRequest | None = None
    reason: str

    @model_validator(mode="after")
    def validate_continuation_shape(self) -> "WorkflowContinuation":
        if self.next_step == "EXECUTE_ACTION":
            if self.action_request is None:
                raise ValueError("Execution continuation requires an exact action request")
            if self.action_request.policy_decision is PolicyDecision.BLOCK:
                raise ValueError("Blocked action cannot be an execution continuation")
        elif self.action_request is not None:
            raise ValueError("Replan/close continuation must not carry an execution request")
        return self
