from typing import Protocol

from src.contracts.payloads import DiagnosisCompleted
from src.contracts.tasks import TaskContext


class RootCauseAnalyzer(Protocol):
    async def diagnose(self, context: TaskContext) -> DiagnosisCompleted: ...


class KnowledgeRetriever(Protocol):
    async def retrieve_evidence_refs(self, query: str, limit: int) -> tuple[str, ...]: ...
