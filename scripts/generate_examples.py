"""Create synthetic v1 fixtures; never use production telemetry or credentials."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.contracts.actions import ActionIntent, ActionRequest, ActionTarget
from src.contracts.enums import ActionType, Environment, PolicyDecision, Priority, Severity
from src.contracts.events import EventEnvelope, ServiceIdentity
from src.contracts.payloads import (
    ActionCompleted,
    AnomalyDetected,
    ApprovalRequested,
    ApprovalResolved,
    DiagnosisCompleted,
    Hypothesis,
    ModelHealthAssessed,
    RecoveryVerified,
    ReleaseAssessment,
    ResourceRecommendation,
    WorkflowContinuation,
)
from src.contracts.topics import TOPICS


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "contracts/examples/v1"
    target.mkdir(parents=True, exist_ok=True)
    intent = ActionIntent(
        action_id="ACT-example-001",
        incident_id="INC-example-001",
        action=ActionType.ROLLBACK,
        environment=Environment.TEST,
        target=ActionTarget(service="fraud-api", namespace="sandbox"),
        parameters={"from_version": "v2", "to_version": "v1", "strategy": "rolling"},
        preconditions=("context_version==1",),
        verification_profile="fraud-api-slo-example",
    )
    request = ActionRequest(
        intent=intent,
        policy_decision=PolicyDecision.BLOCK,
        idempotency_key="INC-example-001:ACT-example-001:v1",
    )
    payloads = (
        AnomalyDetected(
            anomaly_id="ANM-example-001",
            detector="synthetic-threshold",
            anomaly_score=0.9,
            signals={"memory_pct": 94, "error_rate": 0.17},
        ),
        RecoveryVerified(
            verification_id="VER-example-001",
            action_id=intent.action_id,
            window="10m",
            status="INCONCLUSIVE",
            checks={"error_rate": "UNKNOWN"},
        ),
        DiagnosisCompleted(
            diagnosis_id="RCA-example-001",
            hypotheses=(
                Hypothesis(
                    rank=1,
                    code="MEMORY_LEAK_POST_DEPLOY",
                    probability=0.86,
                    evidence_refs=("EVD-synthetic-001",),
                ),
            ),
            primary_root_cause="MEMORY_LEAK_POST_DEPLOY",
            recommended_actions=(intent,),
            rationale_summary="Synthetic fixture: memory growth followed a test deployment",
        ),
        ReleaseAssessment(
            assessment_id="REL-example-001",
            commit="example-only",
            artifact_digest="sha256:" + "0" * 64,
            decision="BLOCK",
            blocking_reasons=("Synthetic evidence is not release authorization",),
            evidence_refs=("EVD-synthetic-001",),
            policy_version="skeleton-deny-all-v1",
        ),
        request,
        ActionCompleted(
            action_id=intent.action_id,
            action_version=1,
            status="BLOCKED",
            pre_state={},
            post_state={},
            execution_evidence_refs=(),
            verification_requested=False,
        ),
        ResourceRecommendation(
            recommendation_id="RES-example-001",
            candidate_action=intent.model_copy(
                update={
                    "action_id": "ACT-example-002",
                    "action": ActionType.SCALE_REPLICAS,
                    "parameters": {"replicas": 4, "min": 2, "max": 8},
                }
            ),
            confidence=0.8,
            observation_window="15m",
            rollback_condition="latency fails to improve",
            constraints=("replicas<=8",),
        ),
        ModelHealthAssessed(
            assessment_id="MHA-example-001",
            model_name="fraud-detection",
            model_version="v1",
            drift_detected=True,
            drift_score=0.37,
            decision="RETRAIN_AND_EVALUATE",
            evidence_refs=("EVD-synthetic-001",),
            dataset_ref="DATA-synthetic-001",
        ),
        ApprovalRequested(
            approval_id="APR-example-001",
            intent=intent,
            action_fingerprint=intent.fingerprint(),
            reason="Synthetic fixture; no executable approval exists",
            expires_at=datetime(2026, 9, 29, 12, 10, tzinfo=UTC),
        ),
        ApprovalResolved(
            approval_id="APR-example-001",
            action_id=intent.action_id,
            action_version=1,
            decision="REJECTED",
            approver_id="synthetic-operator",
            comment="Example only",
        ),
        WorkflowContinuation(
            incident_id=intent.incident_id,
            next_step="REPLAN",
            reason="Synthetic rejection",
        ),
    )
    payload_by_type = {type(payload): payload for payload in payloads}
    for index, (topic, (message_type, model)) in enumerate(TOPICS.items(), start=1):
        source = (
            "orchestrator" if topic.startswith(("approval.", "workflow.")) else "example-fixture"
        )
        event = EventEnvelope(
            message_id=f"MSG-example-{index:03d}",
            correlation_id="INC-example-001",
            trace_id="TRACE-example-001",
            source=source,
            target="orchestrator",
            topic=topic,
            message_type=message_type,
            priority=Priority.P2,
            severity=Severity.MEDIUM,
            timestamp=datetime(2026, 9, 29, 12, tzinfo=UTC),
            environment=Environment.TEST,
            service=ServiceIdentity(service_id="fraud-api", namespace="sandbox", version="v2"),
            evidence_refs=("EVD-synthetic-001",),
            idempotency_key=f"example:{topic}:v1",
            payload=payload_by_type[model].model_dump(mode="json"),
        )
        (target / f"{topic}.json").write_text(
            json.dumps(event.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote {len(TOPICS)} synthetic event examples")


if __name__ == "__main__":
    main()
