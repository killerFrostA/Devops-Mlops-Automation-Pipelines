import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.diagnosis.contracts import DiagnosisAgentResult
from src.agents.diagnosis.file_runner import (
    DiagnosisFileError,
    IncidentFile,
    load_incident_file,
    main,
    render_markdown_report,
    run_incident_file,
)
from src.contracts.enums import ResultStatus

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    ("filename", "expected_cause"),
    [
        ("memory_pressure.json", "MEMORY_PRESSURE"),
        ("post_deploy_errors.json", "DEPENDENCY_UNAVAILABLE"),
        ("contradictory_signals.json", "INSUFFICIENT_EVIDENCE"),
        ("insufficient_evidence.json", "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_file_runner_processes_incident_examples(
    filename: str,
    expected_cause: str,
    tmp_path: Path,
) -> None:
    result, json_path, markdown_path = run_incident_file(EXAMPLES / filename, tmp_path)

    assert isinstance(result, DiagnosisAgentResult)
    assert result.status is ResultStatus.SUCCEEDED
    assert result.diagnosis.primary_root_cause == expected_cause
    assert json_path.exists()
    assert markdown_path.exists()
    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert serialized["diagnosis"]["primary_root_cause"] == expected_cause
    assert result.diagnosis.diagnosis_id == f"DIAG-{result.incident_id}"
    assert [item["code"] for item in serialized["explainability"]["hypothesis_explanations"]] == [
        item.code for item in result.diagnosis.hypotheses
    ]
    assert expected_cause in markdown
    assert "Explicabilité des hypothèses" in markdown


def test_file_runner_preserves_supplied_evidence_and_report_traceability(tmp_path: Path) -> None:
    result, json_path, markdown_path = run_incident_file(
        EXAMPLES / "memory_pressure.json",
        tmp_path,
    )
    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    report = markdown_path.read_text(encoding="utf-8")

    assert "FILE-EVD-MEMORY-001" in report
    assert "92%" in report
    supplied = next(item for item in result.evidence if item.evidence_id == "FILE-EVD-MEMORY-001")
    serialized_supplied = next(
        item for item in serialized["evidence"] if item["evidence_id"] == supplied.evidence_id
    )
    assert serialized_supplied["summary"] == supplied.summary
    assert serialized_supplied["value"] == supplied.value
    assert supplied.evidence_id in serialized["explainability"]["observed_evidence_refs"]
    assert result.explainability.recommendation_explanations
    assert result.explainability.recommendation_explanations[0].action == "COLLECT_MORE_EVIDENCE"
    evidence_ids = {item.evidence_id for item in result.evidence}
    assert set(result.explainability.recommendation_explanations[0].evidence_refs) <= evidence_ids
    assert "Justification de COLLECT_MORE_EVIDENCE" in report
    assert all(
        reference in {item.evidence_id for item in result.evidence}
        for hypothesis in result.diagnosis.hypotheses
        for reference in hypothesis.evidence_refs
    )
    assert "not calibrated probabilities" in report


def test_file_runner_keeps_retrieved_knowledge_separate_from_evidence(tmp_path: Path) -> None:
    result, json_path, markdown_path = run_incident_file(
        EXAMPLES / "memory_pressure.json",
        tmp_path,
    )
    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    report = markdown_path.read_text(encoding="utf-8")
    evidence_ids = {item.evidence_id for item in result.evidence}
    hypothesis_evidence_refs = {
        reference
        for hypothesis in result.diagnosis.hypotheses
        for reference in hypothesis.evidence_refs
    }

    assert result.knowledge_refs
    assert serialized["knowledge_refs"] == list(result.knowledge_refs)
    assert all(reference in report for reference in result.knowledge_refs)
    assert set(result.knowledge_refs).isdisjoint(evidence_ids)
    assert set(result.knowledge_refs).isdisjoint(hypothesis_evidence_refs)
    assert hypothesis_evidence_refs <= evidence_ids


def test_file_runner_ranks_multiple_candidates(tmp_path: Path) -> None:
    result, _, _ = run_incident_file(EXAMPLES / "multiple_causes.json", tmp_path)

    causes = [hypothesis.code for hypothesis in result.diagnosis.hypotheses]
    assert len(causes) == 2
    assert "MEMORY_PRESSURE" in causes
    assert "RESOURCE_PRESSURE_OR_CPU_STARVATION" in causes
    assert [item.rank for item in result.explainability.hypothesis_explanations] == [1, 2]
    assert all(item.ranking_reason for item in result.explainability.hypothesis_explanations)
    assert all(
        set(item.supporting_evidence_refs) <= {evidence.evidence_id for evidence in result.evidence}
        for item in result.explainability.hypothesis_explanations
    )
    markdown = (tmp_path / "multiple_causes.diagnosis.md").read_text(encoding="utf-8")
    for item in result.explainability.hypothesis_explanations:
        assert item.code in markdown
        assert item.ranking_reason in markdown


def test_file_runner_accepts_dated_change_and_memory_rise(tmp_path: Path) -> None:
    incident_path = tmp_path / "dated-memory-incident.json"
    incident_path.write_text(
        json.dumps(
            {
                "incident_id": "INC-DATED-MEMORY",
                "service_id": "payments-api",
                "namespace": "production",
                "environment": "production",
                "issue_summary": "Memory rose with a restart after a dated deployment.",
                "signals": {"memory_percent": 90, "restart_count": 1},
                "memory_rise_started_at": "2026-10-19T12:00:00Z",
                "recent_change_events": [
                    {
                        "change_id": "DEPLOY-26",
                        "description": "Deployment v2.6",
                        "changed_at": "2026-10-19T11:45:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result, json_path, markdown_path = run_incident_file(incident_path, tmp_path)

    assert result.diagnosis.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    change_ref = "EVD-INC-DATED-MEMORY-CHANGE-01"
    assert change_ref in result.diagnosis.hypotheses[0].evidence_refs
    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    assert serialized["diagnosis"]["primary_root_cause"] == "MEMORY_LEAK_POST_DEPLOY"
    assert change_ref in markdown_path.read_text(encoding="utf-8")


def test_contradictory_example_reports_disagreement_and_uncertainty(tmp_path: Path) -> None:
    result, json_path, markdown_path = run_incident_file(
        EXAMPLES / "contradictory_signals.json",
        tmp_path,
    )

    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert result.diagnosis.primary_root_cause == "INSUFFICIENT_EVIDENCE"
    assert result.diagnosis.requires_additional_evidence is True
    assert result.explainability.contradictions
    assert serialized["explainability"]["contradictions"] == list(
        result.explainability.contradictions
    )
    assert "Contradiction" in markdown
    assert "INSUFFICIENT_EVIDENCE" in markdown
    assert result.diagnosis.recommended_actions == ()


def test_invalid_incident_file_reports_field_and_writes_no_reports(tmp_path: Path) -> None:
    with pytest.raises(DiagnosisFileError, match="service_id"):
        run_incident_file(EXAMPLES / "invalid_missing_service.json", tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_file_runner_reports_reference_without_attached_evidence(tmp_path: Path) -> None:
    result, _, markdown_path = run_incident_file(EXAMPLES / "insufficient_evidence.json", tmp_path)

    assert any(
        "UPSTREAM-EVIDENCE-NOT-INCLUDED" in limitation and "not resolved" in limitation
        for limitation in result.diagnosis.limitations
    )
    assert result.diagnosis.primary_root_cause == "INSUFFICIENT_EVIDENCE"
    assert result.diagnosis.requires_additional_evidence is True
    assert any(
        "No memory utilization measurement" in item
        for item in result.explainability.missing_information
    )
    assert "No memory utilization measurement" in markdown_path.read_text(encoding="utf-8")


def test_invalid_json_has_line_and_column_context(tmp_path: Path) -> None:
    invalid_json = tmp_path / "broken.json"
    invalid_json.write_text('{"incident_id": ', encoding="utf-8")

    with pytest.raises(DiagnosisFileError, match=r"line 1, column"):
        load_incident_file(invalid_json)


def test_input_rejects_unknown_fields_and_duplicate_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        IncidentFile.model_validate(
            {
                "incident_id": "INC",
                "service_id": "svc",
                "namespace": "test",
                "environment": "test",
                "issue_summary": "Problem",
                "unexpected": True,
            }
        )

    with pytest.raises(ValidationError, match="unique"):
        IncidentFile.model_validate(
            {
                "incident_id": "INC",
                "service_id": "svc",
                "namespace": "test",
                "environment": "test",
                "issue_summary": "Problem",
                "evidence_refs": ["EVD-1", "EVD-1"],
            }
        )


def test_cli_writes_both_reports_and_prints_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            str(EXAMPLES / "insufficient_evidence.json"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "JSON result:" in output
    assert "Markdown report:" in output
    assert len(list(tmp_path.glob("*.diagnosis.json"))) == 1
    assert len(list(tmp_path.glob("*.diagnosis.md"))) == 1


def test_markdown_report_labels_local_knowledge_as_fixtures(tmp_path: Path) -> None:
    result, _, _ = run_incident_file(EXAMPLES / "memory_pressure.json", tmp_path)

    report = render_markdown_report(result, load_incident_file(EXAMPLES / "memory_pressure.json"))

    assert "pas une recherche Qdrant" in report
    assert "aucun LLM réel" in report
    assert "Incidents passés similaires" in report
