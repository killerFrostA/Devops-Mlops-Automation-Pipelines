from src.contracts.actions import ActionIntent
from src.contracts.base import Contract
from src.contracts.enums import PolicyDecision


class PolicyAssessment(Contract):
    decision: PolicyDecision
    action_fingerprint: str
    reason: str
    policy_version: str = "skeleton-deny-all-v1"


class PolicyService:
    def assess(self, intent: ActionIntent) -> PolicyAssessment:
        return PolicyAssessment(
            decision=PolicyDecision.BLOCK,
            action_fingerprint=intent.fingerprint(),
            reason="Execution policy is not implemented; no action is authorized",
        )
