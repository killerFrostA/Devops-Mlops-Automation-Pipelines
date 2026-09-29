"""Exact action intent; constructing a value never authorizes execution."""

import hashlib
import json
from typing import Annotated

from pydantic import Field, JsonValue, model_validator

from src.contracts.base import Contract
from src.contracts.enums import ActionType, Environment, PolicyDecision


class ActionTarget(Contract):
    service: Annotated[str, Field(min_length=1)]
    namespace: Annotated[str, Field(min_length=1)]


class ActionIntent(Contract):
    action_id: Annotated[str, Field(min_length=1)]
    action_version: Annotated[int, Field(ge=1)] = 1
    incident_id: Annotated[str, Field(min_length=1)]
    action: ActionType
    environment: Environment
    target: ActionTarget
    parameters: dict[str, JsonValue]
    preconditions: tuple[str, ...] = ()
    verification_profile: Annotated[str, Field(min_length=1)]

    def fingerprint(self) -> str:
        """Bind approval to the entire current intent, including its version and scope."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ActionRequest(Contract):
    intent: ActionIntent
    policy_decision: PolicyDecision
    approval_id: str | None = None
    approval_token: Annotated[str | None, Field(repr=False)] = None
    idempotency_key: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def require_approval_proof_shape(self) -> "ActionRequest":
        if self.policy_decision is PolicyDecision.APPROVAL_REQUIRED:
            if not self.approval_id or not self.approval_token:
                raise ValueError("Approval-required requests need an approval ID and token")
        elif self.approval_id is not None or self.approval_token is not None:
            raise ValueError("Approval proof is only valid for approval-required requests")
        # Presence is not verification: the executor must validate scope/signature/expiry.
        return self
