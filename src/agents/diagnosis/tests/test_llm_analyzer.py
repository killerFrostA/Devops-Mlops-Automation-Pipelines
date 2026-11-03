from src.agents.diagnosis.contracts import DiagnosisInput, KnowledgeRetrieval
from src.agents.diagnosis.evidence_collector import collect_evidence
from src.agents.diagnosis.llm_analyzer import analyze_with_llm
from src.agents.diagnosis.rules import evaluate_hypotheses
from src.contracts.enums import Environment


def test_llm_analyzer_returns_structured_summary_for_memory_incident() -> None:
    payload = DiagnosisInput(
        incident_id="INC-001",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory usage is high and pods keep restarting after deployment.",
        signals={"memory_percent": 84.0, "restart_count": 5, "latency_ms": 980},
        recent_changes=["deploy v2.6"],
        evidence=[],
        constraints=["No rollback auto-execution."],
    )

    evidence = tuple(collect_evidence(payload))
    result = analyze_with_llm(
        payload,
        tuple(evaluate_hypotheses(payload.model_copy(update={"evidence": list(evidence)}))),
        evidence,
        knowledge=KnowledgeRetrieval(),
    )

    assert "memory" in result.summary.lower()
    assert result.source == "deterministic_simulation"
    assert result.selected_cause == "MEMORY_PRESSURE"
    assert set(result.evidence_refs) <= {item.evidence_id for item in evidence}
    assert "not a language-model" in result.limitations[0].lower()
