from datetime import UTC, datetime

from src.agents.diagnosis.contracts import DiagnosisInput, EvidenceItem
from src.agents.diagnosis.evidence_collector import collect_evidence
from src.agents.diagnosis.rules import evaluate_hypotheses, signal_conflicts


def test_memory_leak_rule_ranked_above_other_causes() -> None:
    evidence = [
        EvidenceItem(
            evidence_id="EVD-METRIC-01",
            evidence_type="metric",
            source="prometheus",
            observed_at=datetime.now(UTC),
            summary="memory percent climbed from 42 to 84 over 20 minutes",
            value={"memory_percent": 84.0},
            confidence=0.96,
        ),
        EvidenceItem(
            evidence_id="EVD-K8S-01",
            evidence_type="k8s_event",
            source="kubernetes",
            observed_at=datetime.now(UTC),
            summary="pod restart loop triggered after deployment",
            value={"restart_count": 5},
            confidence=0.9,
        ),
        EvidenceItem(
            evidence_id="EVD-GIT-01",
            evidence_type="git_change",
            source="git",
            observed_at=datetime.now(UTC),
            summary="new deployment version v2.6 published 9 minutes ago",
            value={"version": "v2.6"},
            confidence=0.88,
        ),
    ]

    payload = DiagnosisInput(
        incident_id="INC-001",
        service_id="payment-api",
        namespace="prod",
        environment="production",
        issue_summary="Memory usage is high and pods keep restarting after deployment.",
        signals={"memory_percent": 84.0, "restart_count": 5, "latency_ms": 980},
        recent_changes=["deploy v2.6", "config change on limits"],
        evidence=evidence,
        constraints=["Do not execute the rollback automatically."],
    )

    hypotheses = evaluate_hypotheses(payload)
    assert hypotheses[0].code == "MEMORY_PRESSURE"
    assert hypotheses[0].probability >= 0.7
    assert any("MEMORY_PRESSURE" == h.code for h in hypotheses)


def test_insufficient_evidence_keeps_additional_evidence_flag() -> None:
    payload = DiagnosisInput(
        incident_id="INC-002",
        service_id="auth-api",
        namespace="prod",
        environment="production",
        issue_summary="Service is slow but evidence is minimal.",
        signals={"latency_ms": 120},
        evidence=[],
    )

    hypotheses = evaluate_hypotheses(payload)
    assert any(h.code == "INSUFFICIENT_EVIDENCE" for h in hypotheses)
    assert hypotheses[0].probability <= 0.4


def test_rules_return_multiple_ranked_causes_with_cause_relevant_evidence(
    operational_evidence_fixture,
) -> None:
    payload = DiagnosisInput(
        incident_id="INC-MULTI",
        service_id="payment-api",
        namespace="prod",
        environment="production",
        issue_summary="Memory growth, CPU pressure, and restarts followed a deployment.",
        signals={
            "memory_percent": 88,
            "cpu_percent": 92,
            "restart_count": 3,
            "latency_ms": 900,
        },
        recent_changes=["deploy v2.6"],
        evidence=operational_evidence_fixture,
    )

    hypotheses = evaluate_hypotheses(payload.model_copy(update={"evidence": collect_evidence(payload)}))

    assert len(hypotheses) == 2
    assert [item.rank for item in hypotheses] == [1, 2]
    assert hypotheses[0].probability >= hypotheses[1].probability
    memory = next(item for item in hypotheses if item.code == "MEMORY_PRESSURE")
    resources = next(
        item for item in hypotheses if item.code == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    )
    assert "FIX-METRIC-01" in memory.evidence_refs
    assert "FIX-GIT-01" not in memory.evidence_refs
    assert "FIX-K8S-01" not in memory.evidence_refs
    assert "FIX-LOG-01" not in memory.evidence_refs
    assert "FIX-METRIC-01" not in resources.evidence_refs
    assert resources.evidence_refs


def test_conflicting_memory_readings_do_not_support_memory_cause() -> None:
    payload = DiagnosisInput(
        incident_id="INC-CONFLICT",
        service_id="payment-api",
        namespace="prod",
        environment="production",
        issue_summary="Conflicting memory measurements were supplied.",
        signals={"memory_percent": 92, "memory_pct": 28, "restart_count": 4},
        recent_changes=["deploy v2.6"],
    )

    hypotheses = evaluate_hypotheses(payload)

    assert hypotheses[0].code == "INSUFFICIENT_EVIDENCE"


def test_unrelated_evidence_is_not_cited_as_support_for_a_memory_candidate() -> None:
    payload = DiagnosisInput(
        incident_id="INC-TRACE",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="Memory is high and the process restarted.",
        signals={"memory_percent": 88, "restart_count": 3},
        recent_changes=["unrelated config cleanup"],
        evidence=[
            EvidenceItem(
                evidence_id="GIT-UNRELATED",
                evidence_type="git_change",
                source="git",
                summary="A cleanup change unrelated to deployment.",
                value={"change": "cleanup"},
            ),
            EvidenceItem(
                evidence_id="LOG-UNRELATED",
                evidence_type="log",
                source="logs",
                summary="Routine request completed.",
            ),
        ],
    )

    collected = payload.model_copy(update={"evidence": collect_evidence(payload)})
    hypothesis = evaluate_hypotheses(collected)[0]

    assert hypothesis.code == "MEMORY_PRESSURE"
    assert "GIT-UNRELATED" not in hypothesis.evidence_refs
    assert "LOG-UNRELATED" not in hypothesis.evidence_refs
    assert any(reference.endswith("METRIC-01") for reference in hypothesis.evidence_refs)


def test_contradictions_on_cpu_measurement_suppress_resource_candidate() -> None:
    payload = DiagnosisInput(
        incident_id="INC-CPU-CONFLICT",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="CPU measurements disagree.",
        signals={"cpu_percent": 94},
        evidence=[
            EvidenceItem(
                evidence_id="CPU-LOW",
                evidence_type="metric",
                source="manual-export",
                summary="CPU was low in the same window.",
                value={"cpu_percent": 22},
            )
        ],
    )

    hypotheses = evaluate_hypotheses(payload)

    assert hypotheses[0].code == "INSUFFICIENT_EVIDENCE"
    assert any("cpu_percent" in conflict for conflict in signal_conflicts(payload))


def test_conflicted_cpu_evidence_is_not_cited_when_error_rate_supports_candidate() -> None:
    payload = DiagnosisInput(
        incident_id="INC-CPU-ERROR-CONFLICT",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="CPU readings disagree while error rate is high.",
        signals={"cpu_percent": 94, "error_rate": 15},
        evidence=[
            EvidenceItem(
                evidence_id="CPU-LOW",
                evidence_type="metric",
                source="manual-export",
                summary="CPU was low in the same window.",
                value={"cpu_percent": 22},
            )
        ],
    )

    collected = payload.model_copy(update={"evidence": collect_evidence(payload)})
    hypothesis = evaluate_hypotheses(collected)[0]

    assert hypothesis.code == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert "CPU-LOW" not in hypothesis.evidence_refs
    assert any(reference.endswith("METRIC-05") for reference in hypothesis.evidence_refs)


def test_deployment_and_high_memory_without_independent_failure_signal_are_insufficient() -> None:
    payload = DiagnosisInput(
        incident_id="INC-DEPLOY-ONLY",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="Memory is high after a deployment.",
        signals={"memory_percent": 88},
        recent_changes=["deploy version 2.6"],
    )

    assert evaluate_hypotheses(payload)[0].code == "MEMORY_PRESSURE"


def test_memory_leak_requires_high_memory_failure_signal_and_prior_dated_change() -> None:
    payload = DiagnosisInput(
        incident_id="INC-MEMORY-LEAK",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="Memory rose and the workload restarted after a dated deployment.",
        signals={"memory_percent": 88, "restart_count": 1},
        memory_rise_started_at=datetime(2026, 10, 19, 12, 0, tzinfo=UTC),
        recent_change_events=[
            {
                "change_id": "DEPLOY-26",
                "description": "Deployment v2.6",
                "changed_at": datetime(2026, 10, 19, 11, 45, tzinfo=UTC),
            }
        ],
    )

    hypotheses = evaluate_hypotheses(payload.model_copy(update={"evidence": collect_evidence(payload)}))

    assert hypotheses[0].code == "MEMORY_LEAK_POST_DEPLOY"
    assert "EVD-INC-MEMORY-LEAK-CHANGE-01" in hypotheses[0].evidence_refs
    assert any(reference.endswith("METRIC-01") for reference in hypotheses[0].evidence_refs)
    assert any(reference.endswith("METRIC-04") for reference in hypotheses[0].evidence_refs)


def test_memory_leak_boundary_requires_change_strictly_before_rise() -> None:
    rise_started_at = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    for change_at in (
        rise_started_at,
        datetime(2026, 10, 19, 12, 1, tzinfo=UTC),
    ):
        payload = DiagnosisInput(
            incident_id="INC-MEMORY-BOUNDARY",
            service_id="payment-api",
            namespace="test",
            environment="test",
            issue_summary="Memory is high with a restart, but temporal order is not prior.",
            signals={"memory_percent": 88, "restart_count": 1},
            memory_rise_started_at=rise_started_at,
            recent_change_events=[
                {
                    "change_id": "DEPLOY-26",
                    "description": "Deployment v2.6",
                    "changed_at": change_at,
                }
            ],
        )

        hypotheses = evaluate_hypotheses(
            payload.model_copy(update={"evidence": collect_evidence(payload)})
        )

        assert hypotheses[0].code == "MEMORY_PRESSURE"


def test_memory_leak_requires_a_timestamped_rise_even_with_dated_change_and_oom() -> None:
    payload = DiagnosisInput(
        incident_id="INC-MEMORY-NO-RISE-TIME",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="High memory, OOM, and a dated deployment; no rise start time.",
        signals={"memory_percent": 90},
        evidence=[
            EvidenceItem(
                evidence_id="OOM-1",
                evidence_type="k8s_event",
                source="manual",
                summary="Container terminated OOMKilled.",
            )
        ],
        recent_change_events=[
            {
                "change_id": "DEPLOY-27",
                "description": "Deployment v2.7",
                "changed_at": datetime(2026, 10, 19, 11, 45, tzinfo=UTC),
            }
        ],
    )

    assert evaluate_hypotheses(payload)[0].code == "MEMORY_PRESSURE"


def test_high_memory_without_failure_signal_remains_memory_pressure() -> None:
    payload = DiagnosisInput(
        incident_id="INC-MEMORY-HIGH-ONLY",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="Memory is high but no growth, restart, or OOM is reported.",
        signals={"memory_percent": 88},
        memory_rise_started_at=datetime(2026, 10, 19, 12, 0, tzinfo=UTC),
        recent_change_events=[
            {
                "change_id": "DEPLOY-28",
                "description": "Deployment v2.8",
                "changed_at": datetime(2026, 10, 19, 11, 45, tzinfo=UTC),
            }
        ],
    )

    assert evaluate_hypotheses(payload)[0].code == "MEMORY_PRESSURE"


def test_oom_observation_and_prior_change_qualify_without_restart_count() -> None:
    payload = DiagnosisInput(
        incident_id="INC-MEMORY-OOM",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="High memory and OOM event after a dated change.",
        signals={"memory_percent": 91},
        memory_rise_started_at=datetime(2026, 10, 19, 12, 0, tzinfo=UTC),
        recent_change_events=[
                {
                    "change_id": "DEPLOY-LIMITS-CHANGE",
                    "description": "Deployment changed the container memory limit",
                    "changed_at": datetime(2026, 10, 19, 11, 30, tzinfo=UTC),
                }
            ],
        evidence=[
            EvidenceItem(
                evidence_id="OOM-2",
                evidence_type="k8s_event",
                source="manual",
                summary="Container terminated OOMKilled.",
            )
        ],
    )

    assert evaluate_hypotheses(payload)[0].code == "MEMORY_LEAK_POST_DEPLOY"


def test_recorded_memory_growth_and_prior_change_qualify_without_restart_or_oom() -> None:
    payload = DiagnosisInput(
        incident_id="INC-MEMORY-GROWTH",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="Memory growth followed a dated change.",
        signals={"memory_percent": 88},
        memory_rise_started_at=datetime(2026, 10, 19, 12, 0, tzinfo=UTC),
        memory_growth_observed=True,
        recent_change_events=[
            {
                "change_id": "DEPLOY-GROWTH",
                "description": "Deployment v2.10",
                "changed_at": datetime(2026, 10, 19, 11, 45, tzinfo=UTC),
            }
        ],
    )

    collected = payload.model_copy(update={"evidence": collect_evidence(payload)})
    hypothesis = evaluate_hypotheses(collected)[0]

    assert hypothesis.code == "MEMORY_LEAK_POST_DEPLOY"
    assert "EVD-INC-MEMORY-GROWTH-MEMORY-GROWTH" in hypothesis.evidence_refs


def test_explicit_absence_of_oom_does_not_count_as_failure_indicator() -> None:
    payload = DiagnosisInput(
        incident_id="INC-NO-OOM",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="High memory but no OOM was observed.",
        signals={"memory_percent": 88},
        memory_rise_started_at=datetime(2026, 10, 19, 12, 0, tzinfo=UTC),
        recent_change_events=[
            {
                "change_id": "DEPLOY-NO-OOM",
                "description": "Deployment v2.9",
                "changed_at": datetime(2026, 10, 19, 11, 45, tzinfo=UTC),
            }
        ],
        evidence=[
            EvidenceItem(
                evidence_id="NO-OOM",
                evidence_type="log",
                source="manual",
                summary="No OOM event was observed during the incident.",
            )
        ],
    )

    assert evaluate_hypotheses(payload)[0].code == "MEMORY_PRESSURE"
