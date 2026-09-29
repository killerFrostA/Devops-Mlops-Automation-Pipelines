from typing import Protocol

from src.contracts.actions import ActionRequest
from src.contracts.payloads import ActionCompleted


class DeploymentBackend(Protocol):
    """Validate policy/token/context/idempotency before any external write."""

    async def execute(self, request: ActionRequest) -> ActionCompleted: ...
