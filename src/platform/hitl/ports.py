from typing import Literal, Protocol

from src.contracts.actions import ActionIntent
from src.contracts.payloads import ApprovalRequested, ApprovalResolved


class HITLService(Protocol):
    async def create_approval(self, intent: ActionIntent, reason: str) -> ApprovalRequested: ...

    async def resolve_approval(
        self,
        approval_id: str,
        decision: Literal["APPROVE", "REJECT"],
        authenticated_actor_id: str,
        comment: str,
    ) -> ApprovalResolved: ...

    async def verify_execution_token(self, token: str, intent: ActionIntent) -> bool: ...
