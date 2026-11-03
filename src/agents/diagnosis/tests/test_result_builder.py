from src.agents.diagnosis.contracts import (
    DiagnosisInput,
    EvidenceItem,
    Hypothesis,
    KnowledgeRetrieval,
    SupplementalAnalysis,
)
from src.agents.diagnosis.result_builder import build_result
from src.contracts.enums import Environment


class FailingRetriever:
    def retrieve(self, query: str, limit: int) -> KnowledgeRetrieval:
        raise RuntimeError(f"offline fixture failure for {query!r} with limit {limit}")


class InvalidAnalyzer:
    def analyze(
        self,
        payload: DiagnosisInput,
        hypotheses: tuple[Hypothesis, ...],
        evidence: tuple[EvidenceItem, ...],
        knowledge: KnowledgeRetrieval,
    ) -> dict[str, str]:
        return {
            "source": "untrusted",
            "summary": (
                f"Unsupported text for {payload.incident_id}, {len(hypotheses)} candidates, "
                f"{len(evidence)} evidence items, and "
                f"{len(knowledge.similar_incidents)} similar incidents"
            ),
            "selected_cause": "UNSUPPORTED_CAUSE",
            "evidence_refs": "not-a-tuple",
        }


class FailingAnalyzer:
    def analyze(
        self,
        payload: DiagnosisInput,
        hypotheses: tuple[Hypothesis, ...],
        evidence: tuple[EvidenceItem, ...],
        knowledge: KnowledgeRetrieval,
    ) -> SupplementalAnalysis:
        evidence_count = len(evidence)
        candidate_count = len(hypotheses)
        retrieved_count = len(knowledge.similar_incidents) + len(knowledge.runbooks)
        raise RuntimeError(
            f"offline fixture failure for {payload.incident_id}: "
            f"{candidate_count} candidates, {evidence_count} evidence items, "
            f"{retrieved_count} knowledge items"
        )


def test_build_result_returns_primary_root_cause_and_ranked_hypotheses() -> None:
    payload = DiagnosisInput(
        incident_id="INC-001",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory usage is high and pods keep restarting after deployment.",
        signals={"memory_percent": 84.0, "restart_count": 5, "latency_ms": 980},
        recent_changes=["deploy v2.6"],
        evidence=[],
        constraints=["No rollback auto execution."],
    )

    result = build_result(payload)

    assert result.primary_root_cause == "MEMORY_PRESSURE"
    assert result.hypotheses[0].rank == 1
    assert result.hypotheses[0].probability >= 0.7
    assert result.rationale_summary
    assert result.knowledge_refs


def test_build_result_degrades_with_explicit_limits_when_adapters_fail() -> None:
    payload = DiagnosisInput(
        incident_id="INC-ADAPTER-FAILURE",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory usage is high and pods keep restarting after deployment.",
        signals={"memory_percent": 84.0, "restart_count": 5},
        recent_changes=["deploy v2.6"],
    )

    result = build_result(payload, FailingRetriever(), FailingAnalyzer())

    assert result.primary_root_cause == "MEMORY_PRESSURE"
    assert any("Knowledge retrieval failed" in item for item in result.limitations)
    assert any("Supplemental analysis failed" in item for item in result.limitations)


def test_invalid_supplemental_output_cannot_change_rule_candidates() -> None:
    payload = DiagnosisInput(
        incident_id="INC-INVALID-ANALYZER",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="CPU is saturated and latency is high.",
        signals={"cpu_percent": 90, "latency_ms": 900},
    )

    result = build_result(payload, analyzer=InvalidAnalyzer())  # type: ignore[arg-type]

    assert result.primary_root_cause == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert all(item.code != "UNSUPPORTED_CAUSE" for item in result.hypotheses)
    assert any("Supplemental analysis failed" in item for item in result.limitations)


def test_memory_candidate_without_deployment_does_not_recommend_rollback() -> None:
    payload = DiagnosisInput(
        incident_id="INC-NO-DEPLOY",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory is high and the workload restarted.",
        signals={"memory_percent": 88, "restart_count": 3},
        recent_changes=["unrelated config cleanup"],
    )

    result = build_result(payload)

    assert result.primary_root_cause == "MEMORY_PRESSURE"
    assert [item.action for item in result.recommended_actions] == ["COLLECT_MORE_EVIDENCE"]
    assert result.requires_additional_evidence is True
    assessment = result.explainability.hypothesis_explanations[0]
    assert any("not timestamped" in item.lower() for item in assessment.missing_information)
    assert not any(
        item.action == "ROLLBACK_OR_REVERT_DEPLOYMENT" for item in result.recommended_actions
    )


def test_coincident_deployment_is_not_enough_to_recommend_rollback() -> None:
    payload = DiagnosisInput(
        incident_id="INC-DEPLOY-COINCIDENCE",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory is high and the workload restarted after a deployment.",
        signals={"memory_percent": 88, "restart_count": 3},
        recent_changes=["deploy version 2.6"],
    )

    result = build_result(payload)

    assert result.primary_root_cause == "MEMORY_PRESSURE"
    assert [item.action for item in result.recommended_actions] == ["COLLECT_MORE_EVIDENCE"]
    assert result.requires_additional_evidence is True
    assert any("complete dated evidence" in item.lower() for item in result.limitations)


def test_explainability_separates_provided_derived_and_contradictory_data() -> None:
    payload = DiagnosisInput(
        incident_id="INC-EXPLAIN",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory readings disagree.",
        signals={"memory_percent": 90, "restart_count": 3},
        evidence=[
            EvidenceItem(
                evidence_id="MEMORY-LOW",
                evidence_type="metric",
                source="manual-export",
                summary="Memory measured at 30%.",
                value={"memory_percent": 30},
            )
        ],
    )

    result = build_result(payload)

    assert result.primary_root_cause == "INSUFFICIENT_EVIDENCE"
    assert result.explainability.observed_evidence_refs == ("MEMORY-LOW",)
    assert result.explainability.derived_evidence_refs
    assert any("memory_percent" in item for item in result.explainability.contradictions)
    assert result.requires_additional_evidence is True


def test_non_numeric_evidence_does_not_count_as_a_measured_signal() -> None:
    payload = DiagnosisInput(
        incident_id="INC-NONNUMERIC",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="CPU value is not numeric.",
        evidence=[
            EvidenceItem(
                evidence_id="CPU-INVALID",
                evidence_type="metric",
                source="manual-export",
                summary="CPU field was not parsed.",
                value={"cpu_percent": "high"},
            )
        ],
    )

    result = build_result(payload)

    assert result.primary_root_cause == "INSUFFICIENT_EVIDENCE"
    assert any(
        "No CPU utilization measurement was supplied" in item
        for item in result.explainability.missing_information
    )
