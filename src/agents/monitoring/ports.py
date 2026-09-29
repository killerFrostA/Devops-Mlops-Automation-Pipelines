from typing import Protocol

from src.contracts.payloads import AnomalyDetected, RecoveryVerified
from src.contracts.tasks import TaskContext


class AnomalyDetector(Protocol):
    async def detect(self, context: TaskContext) -> AnomalyDetected | None: ...


class RecoveryVerifier(Protocol):
    async def verify(self, action_id: str, context: TaskContext) -> RecoveryVerified: ...
