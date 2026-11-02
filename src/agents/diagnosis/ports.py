from typing import Protocol

from src.agents.diagnosis.contracts import (
    DiagnosisInput,
    EvidenceItem,
    Hypothesis,
    KnowledgeRetrieval,
    SupplementalAnalysis,
)


class KnowledgeRetriever(Protocol):
    """Boundary for local or future external incident/runbook retrieval."""

    def retrieve(self, query: str, limit: int) -> KnowledgeRetrieval: ...


class SupplementalAnalyzer(Protocol):
    """Boundary for a future model; implementations must not execute infrastructure actions."""

    def analyze(
        self,
        payload: DiagnosisInput,
        hypotheses: tuple[Hypothesis, ...],
        evidence: tuple[EvidenceItem, ...],
        knowledge: KnowledgeRetrieval,
    ) -> SupplementalAnalysis: ...
