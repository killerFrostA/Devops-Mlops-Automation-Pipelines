import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import JsonValue

from src.agents.diagnosis.contracts import (
    DiagnosisAgentResult,
    DiagnosisInput,
    EvidenceItem,
    Hypothesis,
    KnowledgeRetrieval,
    SupplementalAnalysis,
)
from src.agents.diagnosis.evidence import ResolvedEvidence
from src.agents.diagnosis.service import DiagnosisAgent
from src.contracts.enums import ActionType, AgentId, Environment, ResultStatus
from src.contracts.payloads import DiagnosisCompleted
from src.contracts.tasks import AgentTask, TaskContext


def _task(
    *,
    signals: dict[str, JsonValue] | None = None,
    evidence_refs: tuple[str, ...] = (),
    agent_id: AgentId = AgentId.DIAGNOSIS,
    deadline: datetime | None = None,
) -> AgentTask:
    return AgentTask(
        task_id="TASK-LOCAL-RCA",
        incident_id="INC-LOCAL-RCA",
        agent_id=agent_id,
        objective="Memory and CPU pressure increased after the recent deployment.",
        task_context=TaskContext(
            service_id="payment-api",
            namespace="test",
            environment=Environment.TEST,
            signals=(
                signals
                if signals is not None
                else {
                    "memory_percent": 88,
                    "cpu_percent": 70,
                    "restart_count": 3,
                    "latency_ms": 850,
                    "recent_changes": ["deploy v2.6"],
                }
            ),
            evidence_refs=evidence_refs,
        ),
        deadline=deadline or datetime.now(UTC) + timedelta(minutes=5),
    )


class FailingRetriever:
    def retrieve(self, query: str, limit: int) -> KnowledgeRetrieval:
        raise RuntimeError(f"offline test retriever failure for {query!r} with limit {limit}")


class FailingAnalyzer:
    def analyze(
        self,
        payload: DiagnosisInput,
        hypotheses: tuple[Hypothesis, ...],
        evidence: tuple[EvidenceItem, ...],
        knowledge: KnowledgeRetrieval,
    ) -> SupplementalAnalysis:
        raise RuntimeError(
            f"offline test analyzer failure for {payload.incident_id} with "
            f"{len(hypotheses)} candidates, {len(evidence)} evidence items, and "
            f"{len(knowledge.similar_incidents)} similar fixtures"
        )


def test_service_runs_the_complete_local_diagnosis_path() -> None:
    result = asyncio.run(DiagnosisAgent().handle(_task()))

    assert isinstance(result, DiagnosisAgentResult)
    assert result.status is ResultStatus.SUCCEEDED
    assert isinstance(result.diagnosis, DiagnosisCompleted)
    assert result.diagnosis.primary_root_cause == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert [item.rank for item in result.diagnosis.hypotheses] == [1, 2]
    assert result.diagnosis.recommended_actions[0].action is ActionType.SCALE_REPLICAS
    # CPU-backed scaling is actionable, so the diagnosis is conclusive enough to act on.
    assert result.diagnosis.requires_additional_evidence is False
    assert all(result.evidence_refs)
    assert result.knowledge_refs
    assert result.evidence
    assert all(item.source == "diagnosis_input" for item in result.evidence)
    assert all(item.provenance == "derived" for item in result.evidence)
    assert all(item.observed_at is None and item.confidence is None for item in result.evidence)
    evidence_ids = {item.evidence_id for item in result.evidence}
    assert all(
        reference in evidence_ids
        for hypothesis in result.diagnosis.hypotheses
        for reference in hypothesis.evidence_refs
    )
    assert result.confidence == result.diagnosis.hypotheses[0].probability
    assert result.diagnosis.limitations
    assert any("does not specify a replica count" in item for item in result.diagnosis.limitations)


def test_service_is_reproducible_for_identical_task_input() -> None:
    agent = DiagnosisAgent()
    task = _task()

    first = asyncio.run(agent.handle(task))
    second = asyncio.run(agent.handle(task))

    assert isinstance(first, DiagnosisAgentResult)
    assert isinstance(second, DiagnosisAgentResult)
    assert first.diagnosis == second.diagnosis
    assert first.evidence_refs == second.evidence_refs


def test_service_degrades_to_rules_when_local_adapters_fail() -> None:
    result = asyncio.run(DiagnosisAgent(FailingRetriever(), FailingAnalyzer()).handle(_task()))

    assert isinstance(result, DiagnosisAgentResult)
    assert result.status is ResultStatus.SUCCEEDED
    assert result.diagnosis.primary_root_cause == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert any("Knowledge retrieval failed" in item for item in result.diagnosis.limitations)
    assert any("Supplemental analysis failed" in item for item in result.diagnosis.limitations)


def test_service_returns_insufficient_evidence_without_inventing_proofs() -> None:
    result = asyncio.run(
        DiagnosisAgent().handle(_task(signals={}, evidence_refs=("EVD-UNRESOLVED-1",)))
    )

    assert isinstance(result, DiagnosisAgentResult)
    assert result.diagnosis.primary_root_cause == "INSUFFICIENT_EVIDENCE"
    assert result.diagnosis.hypotheses[0].evidence_refs == ()
    assert result.diagnosis.recommended_actions == ()
    assert result.diagnosis.requires_additional_evidence is True
    assert any(
        "EVD-UNRESOLVED-1" in item and "not resolved" in item
        for item in result.diagnosis.limitations
    )
    assert any("EVD-UNRESOLVED-1" in item for item in result.explainability.missing_information)


def test_service_uses_resolved_evidence_and_preserves_hypothesis_traceability() -> None:
    observed_at = datetime.now(UTC) - timedelta(minutes=2)
    evidence = ResolvedEvidence(
        evidence_id="EVD-RESOLVED-METRIC-01",
        incident_id="INC-LOCAL-RCA",
        evidence_type="metric",
        source="monitoring-agent",
        observed_at=observed_at,
        summary="Memory utilization was 88% and the workload restarted three times.",
        value={"memory_percent": 88, "restart_count": 3},
    )
    task = _task(signals={}, evidence_refs=(evidence.evidence_id,))

    result = asyncio.run(DiagnosisAgent().handle(task, (evidence,)))

    assert isinstance(result, DiagnosisAgentResult)
    assert result.status is ResultStatus.SUCCEEDED
    assert result.diagnosis.primary_root_cause == "MEMORY_PRESSURE"
    assert result.evidence[0].evidence_id == evidence.evidence_id
    assert result.evidence[0].source == "monitoring-agent"
    assert result.evidence[0].provenance == "provided"
    assert result.evidence[0].observed_at == observed_at
    assert evidence.evidence_id in result.diagnosis.hypotheses[0].evidence_refs
    assert evidence.evidence_id in result.evidence_refs
    assert not any("not resolved" in item for item in result.diagnosis.limitations)


def test_service_reports_unresolved_reference_alongside_resolved_evidence() -> None:
    evidence = ResolvedEvidence(
        evidence_id="EVD-RESOLVED-METRIC-02",
        incident_id="INC-LOCAL-RCA",
        evidence_type="metric",
        source="monitoring-agent",
        observed_at=datetime.now(UTC),
        summary="Memory utilization was 88% and the workload restarted three times.",
        value={"memory_percent": 88, "restart_count": 3},
    )
    result = asyncio.run(
        DiagnosisAgent().handle(
            _task(signals={}, evidence_refs=(evidence.evidence_id, "EVD-UNKNOWN")),
            (evidence,),
        )
    )

    assert isinstance(result, DiagnosisAgentResult)
    assert result.diagnosis.primary_root_cause == "MEMORY_PRESSURE"
    assert evidence.evidence_id in result.diagnosis.hypotheses[0].evidence_refs
    assert any(
        "EVD-UNKNOWN" in item and "not resolved" in item for item in result.diagnosis.limitations
    )
    assert result.diagnosis.requires_additional_evidence is True
    assert any("EVD-UNKNOWN" in item for item in result.explainability.missing_information)


def test_service_uses_timestamped_change_metadata_for_memory_taxonomy() -> None:
    result = asyncio.run(
        DiagnosisAgent().handle(
            _task(
                signals={
                    "memory_percent": 90,
                    "restart_count": 1,
                    "memory_rise_started_at": "2026-10-19T12:00:00Z",
                    "recent_change_events": [
                        {
                            "change_id": "DEPLOY-SVC",
                            "description": "Deployment v2.6",
                            "changed_at": "2026-10-19T11:45:00Z",
                        }
                    ],
                }
            )
        )
    )

    assert isinstance(result, DiagnosisAgentResult)
    assert result.diagnosis.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    assert "EVD-INC-LOCAL-RCA-CHANGE-01" in result.diagnosis.hypotheses[0].evidence_refs


def test_task_rejects_unrequested_or_cross_incident_resolved_evidence() -> None:
    evidence = ResolvedEvidence(
        evidence_id="EVD-UNREQUESTED",
        incident_id="INC-LOCAL-RCA",
        evidence_type="log",
        source="monitoring-agent",
        observed_at=datetime.now(UTC),
        summary="A resolved log observation.",
        value={"message": "restart observed"},
    )

    with pytest.raises(ValueError, match="requested by the task"):
        asyncio.run(DiagnosisAgent().handle(_task(), (evidence,)))

    with pytest.raises(ValueError, match="task incident"):
        asyncio.run(
            DiagnosisAgent().handle(
                _task(evidence_refs=(evidence.evidence_id,)),
                (evidence.model_copy(update={"incident_id": "INC-OTHER"}),),
            )
        )


def test_service_reports_invalid_signals_as_failed_result() -> None:
    result = asyncio.run(DiagnosisAgent().handle(_task(signals={"cpu_percent": "high"})))

    assert result.status is ResultStatus.FAILED
    assert "signals" in result.rationale_summary
    assert result.confidence is None


def test_service_reports_invalid_recent_changes_as_failed_result() -> None:
    result = asyncio.run(DiagnosisAgent().handle(_task(signals={"recent_changes": ["deploy", 12]})))

    assert result.status is ResultStatus.FAILED
    assert "recent_changes" in result.rationale_summary


def test_service_preserves_identity_and_deadline_guards() -> None:
    agent = DiagnosisAgent()
    wrong_agent_task = _task(agent_id=AgentId.MONITORING)
    expired_task = _task(deadline=datetime.now(UTC) - timedelta(seconds=1))

    with pytest.raises(ValueError, match="different agent"):
        asyncio.run(agent.handle(wrong_agent_task))
    with pytest.raises(ValueError, match="expired"):
        asyncio.run(agent.handle(expired_task))
