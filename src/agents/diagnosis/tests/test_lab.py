import json
from datetime import UTC, datetime, timedelta

import pytest

from src.agents.diagnosis.agent1_runner import diagnose_agent1_event
from src.agents.diagnosis.evaluation.evaluate import MODES, Scenario, evaluate_scenario
from src.agents.diagnosis.lab import lab
from src.agents.diagnosis.rag_retriever import load_corpus
from src.agents.diagnosis.lab.app import app
from src.agents.diagnosis.signatures import dependency_of, line_signatures
from src.agents.diagnosis.tools import ToolContext, _log_evidence
from src.contracts.events import EventEnvelope

DETECTED = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)


def test_lab_event_is_a_valid_agent1_envelope_that_agent2_diagnoses():
    signals = {"memory_pct": 86.0, "memory_slope_mb_min": 7.5, "pod_restarts": 0.0,
               "error_rate": 0.0, "latency_p95_ms": 30.0}
    raw = lab.build_agent1_event(
        "leak", signals, DETECTED, ("payment-service-rev2", DETECTED - timedelta(minutes=9))
    )

    received = EventEnvelope.model_validate(raw)
    result = diagnose_agent1_event(received)[0].result

    assert raw["payload"]["signals"]["change_context"]["minutes_since_deployment"] == 9.0
    assert result.diagnosis.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"


def test_lab_event_without_change_context_leaves_the_rollout_to_the_tools():
    raw = lab.build_agent1_event("pool", {"error_rate": 0.4}, DETECTED)
    assert "change_context" not in raw["payload"]["signals"]
    assert EventEnvelope.model_validate(raw).severity.value == "HIGH"


def test_captured_lab_incident_becomes_a_replayable_scenario_and_a_rag_document(tmp_path):
    signals = {"memory_pct": 82.0, "memory_slope_mb_min": 6.0, "pod_restarts": 0.0}
    event = lab.build_agent1_event("leak", signals, DETECTED)
    truth = {"fault": "leak", "description": lab.FAULTS["leak"].description,
             "injected_at": (DETECTED - timedelta(minutes=8)).isoformat(),
             "expected_root_cause": "MEMORY_LEAK_POST_DEPLOY",
             "expected_first_action": "ROLLBACK_OR_REVERT_DEPLOYMENT"}
    responses = {"kubernetes": {"deployment_history": [{
        "evidence_type": "k8s_event", "source": "kubernetes",
        "summary": "Deployment revision 2 rolled out.",
        "value": {"change_id": "payment-service-rev2", "description": "Deployment revision 2",
                  "changed_at": (DETECTED - timedelta(minutes=8)).isoformat()},
    }]}}

    raw = lab.scenario_from_lab(lab._next_scenario_id("leak", tmp_path), event, truth, responses)
    scenario = Scenario.model_validate(raw)
    outcome = evaluate_scenario(scenario, MODES[1], None)

    assert scenario.scenario_id == "S01-lab-leak"
    assert outcome.predicted == "MEMORY_LEAK_POST_DEPLOY" and outcome.action_correct
    document = lab.rag_document_from_lab(event, truth)
    for name, documents in (("incidents.json", []), ("runbooks.json", []), ("incidents_lab.json", [document])):
        (tmp_path / name).write_text(json.dumps({"format_version": 1, "documents": documents}), encoding="utf-8")
    assert document["reference_id"].startswith("LAB-LEAK-") and "leak" in document["tags"]
    incidents, _ = load_corpus(tmp_path)
    assert any(item.reference_id == document["reference_id"] for item in incidents)


def test_anomaly_conditions():
    assert lab.anomaly_present(lab.FAULTS["leak"], {"memory_pct": 80})
    assert lab.anomaly_present(lab.FAULTS["leak"], {"memory_pct": 40, "pod_restarts": 1})
    assert not lab.anomaly_present(lab.FAULTS["bad-config"], {"pod_restarts": 1})
    assert lab.anomaly_present(lab.FAULTS["dependency-down"], {"error_rate": 0.5})
    assert not lab.anomaly_present(lab.FAULTS["pool"], {"error_rate": 0.01})


@pytest.mark.parametrize(
    ("line", "signature"),
    [
        ("FATAL invalid configuration: PAYMENT_TIMEOUT_MS='5s' is not a valid positive integer", "config_invalid"),
        ("ERROR payment failed: dial tcp ledger-db:5432: connect: connection refused", "connection_refused"),
        ("ERROR payment failed: connect timed out to ledger-db:5432", "dependency_timeout"),
        ("ERROR could not obtain a connection: connection pool exhausted (size=1)", "pool_exhausted"),
    ],
)
def test_lab_app_log_lines_match_agent2_signatures(line, signature):
    assert signature in line_signatures(line)


def test_probe_failures_of_the_service_itself_are_not_dependency_failures():
    # Seen live during the leak rollout: the new pod was not ready yet.
    probe = 'Readiness probe failed: Get "http://10.244.0.14:8080/healthz": dial tcp 10.244.0.14:8080: connect: connection refused'
    assert line_signatures(probe) == frozenset()
    assert "connection_refused" in line_signatures("dial tcp ledger-db:5432: connect: connection refused")


def test_lab_app_names_the_dependency_and_rejects_bad_config(monkeypatch, capsys):
    context = ToolContext("INC-LAB", "payment-service", "prod", deadline=DETECTED + timedelta(hours=1))
    line = "ERROR payment failed: dial tcp ledger-db:5432: connect: connection refused"
    assert dependency_of(_log_evidence(context, "loki", [line], source="loki")) == "ledger-db"
    monkeypatch.setenv("PAYMENT_TIMEOUT_MS", "5s")
    with pytest.raises(SystemExit):
        app._int_env("PAYMENT_TIMEOUT_MS", 500)
    assert "invalid configuration" in capsys.readouterr().out
