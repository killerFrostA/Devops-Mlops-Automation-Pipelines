from typing import Protocol

from src.contracts.payloads import ReleaseAssessment


class ReleaseAssessor(Protocol):
    async def assess(self, commit: str, artifact_digest: str) -> ReleaseAssessment: ...
