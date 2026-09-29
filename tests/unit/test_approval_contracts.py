import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.contracts.actions import ActionIntent, ActionRequest
from src.contracts.enums import PolicyDecision
from src.contracts.payloads import ApprovalRequested, WorkflowContinuation

ROOT = Path(__file__).resolve().parents[2]


def intent() -> ActionIntent:
    example = ROOT / "contracts/examples/v1/deployment.action.requested.json"
    return ActionIntent.model_validate(json.loads(example.read_text())["payload"]["intent"])


def test_approval_request_rejects_changed_intent() -> None:
    example = ROOT / "contracts/examples/v1/approval.requested.json"
    payload = json.loads(example.read_text())["payload"]
    payload["intent"]["parameters"]["to_version"] = "changed-after-approval"
    with pytest.raises(ValidationError, match="fingerprint"):
        ApprovalRequested.model_validate(payload)


def test_approval_required_request_cannot_omit_proof() -> None:
    with pytest.raises(ValidationError, match="approval ID and token"):
        ActionRequest(
            intent=intent(),
            policy_decision=PolicyDecision.APPROVAL_REQUIRED,
            idempotency_key="test-action",
        )


def test_continuation_cannot_execute_blocked_or_unspecified_action() -> None:
    blocked = ActionRequest(
        intent=intent(), policy_decision=PolicyDecision.BLOCK, idempotency_key="test-action"
    )
    for action_request in (None, blocked):
        with pytest.raises(ValidationError):
            WorkflowContinuation(
                incident_id="INC-test",
                next_step="EXECUTE_ACTION",
                action_request=action_request,
                reason="test",
            )
    with pytest.raises(ValidationError, match="must not carry"):
        WorkflowContinuation(
            incident_id="INC-test", next_step="REPLAN", action_request=blocked, reason="test"
        )
