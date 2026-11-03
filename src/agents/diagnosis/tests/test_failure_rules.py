from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.agents.diagnosis.contracts import (
    DiagnosisInput,
    EvidenceItem,
    EvidenceType,
    RecentChangeEvent,
)
from src.agents.diagnosis.result_builder import build_result, evidence_strength
from src.agents.diagnosis.rules import evaluate_hypotheses
from src.agents.diagnosis.signatures import line_signatures, memory_limit_lowered, signatures_of
from src.contracts.enums import Environment


def _item(
    evidence_id: str, evidence_type: EvidenceType, summary: str, **value: object
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        source="test",
        summary=summary,
        value=value,
    )


def _payload(evidence: list[EvidenceItem], **kwargs: Any) -> DiagnosisInput:
    return DiagnosisInput(
        incident_id="INC-RULES",
        service_id="orders-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Service degraded.",
        evidence=evidence,
        **kwargs,
    )


def test_signatures_detect_known_failures_and_ignore_negations() -> None:
    assert line_signatures("dial tcp db:5432: connect: connection refused") == {
        "connection_refused"
    }
    assert "pool_exhausted" in line_signatures("HikariPool-1 - connection pool exhausted")
    assert "config_invalid" in line_signatures(
        "FATAL invalid configuration: missing required env X"
    )
    assert "crash_loop" in line_signatures("Back-off restarting failed container")
    assert line_signatures("no OOMKilled event was observed") == frozenset()
    assert signatures_of(_item("E", "log", "lines", signatures=["oom", "made_up"])) == {"oom"}


def test_crash_loop_with_invalid_config_log_is_a_configuration_error() -> None:
    hypotheses = evaluate_hypotheses(
        _payload(
            [
                _item(
                    "K8S",
                    "k8s_event",
                    "Pods restarting",
                    reason="CrashLoopBackOff",
                    restart_count=7,
                ),
                _item(
                    "LOG",
                    "log",
                    "Startup failed",
                    lines=["invalid configuration: missing required env DB"],
                ),
            ]
        )
    )

    assert hypotheses[0].code == "CONFIG_ERROR_CRASHLOOP"
    assert hypotheses[0].probability == 0.85
    assert set(hypotheses[0].evidence_refs) == {"K8S", "LOG"}


def test_crash_loop_after_config_change_only_and_unexplained_crash_loop() -> None:
    crash = _item("K8S", "k8s_event", "Pods restarting", reason="CrashLoopBackOff")
    config_change = _item("GIT", "git_change", "Commit", description="Update configmap values.yaml")

    assert evaluate_hypotheses(_payload([crash, config_change]))[0].code == "CONFIG_ERROR_CRASHLOOP"
    assert evaluate_hypotheses(_payload([crash]))[0].code == "CRASH_LOOP_UNEXPLAINED"


def test_oom_crash_loop_is_left_to_the_memory_rules() -> None:
    hypotheses = evaluate_hypotheses(
        _payload(
            [_item("K8S", "k8s_event", "Pods restarting", reason="CrashLoopBackOff OOMKilled")],
            signals={"memory_percent": 97, "restart_count": 5},
        )
    )

    assert all(
        item.code not in ("CONFIG_ERROR_CRASHLOOP", "CRASH_LOOP_UNEXPLAINED") for item in hypotheses
    )


def test_dependency_and_pool_signatures_outrank_symptom_only_cpu_pressure() -> None:
    dependency = evaluate_hypotheses(
        _payload(
            [
                _item(
                    "LOG",
                    "log",
                    "Errors",
                    lines=["connect: connection refused"],
                    dependency="payments-db",
                )
            ],
            signals={"error_rate": 0.25, "latency_ms": 2400},
        )
    )
    pool = evaluate_hypotheses(
        _payload(
            [_item("LOG", "log", "Errors", lines=["too many connections"])],
            signals={"error_rate": 0.08, "latency_ms": 1800},
        )
    )

    assert dependency[0].code == "DEPENDENCY_UNAVAILABLE"
    assert dependency[1].code == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert dependency[1].probability < dependency[0].probability
    assert pool[0].code == "CONNECTION_POOL_EXHAUSTION"


def test_agent1_suspect_alone_gives_a_weak_dependency_hypothesis() -> None:
    hypotheses = evaluate_hypotheses(
        _payload(
            [],
            signals={
                "error_rate": 0.12,
                "agent1_context": {
                    "suspects": [
                        {"service_id": "orders-api", "probability": 0.9},
                        {"service_id": "payments-db", "probability": 0.4},
                    ]
                },
            },
        )
    )

    assert any(
        item.code == "DEPENDENCY_UNAVAILABLE" and item.probability == 0.5 for item in hypotheses
    )


def test_traffic_spike_needs_a_request_rate_baseline_ratio() -> None:
    spike = evaluate_hypotheses(
        _payload(
            [_item("RPS", "metric", "Traffic", rps=1500, rps_baseline=700)],
            signals={"cpu_percent": 93},
        )
    )
    steady = evaluate_hypotheses(
        _payload(
            [_item("RPS", "metric", "Traffic", rps=720, rps_baseline=700)],
            signals={"cpu_percent": 93},
        )
    )

    assert "TRAFFIC_SPIKE" in [item.code for item in spike]
    assert "TRAFFIC_SPIKE" not in [item.code for item in steady]


def test_deployment_older_than_the_correlation_window_is_not_a_post_deploy_leak() -> None:
    rise = datetime(2026, 10, 20, 12, tzinfo=UTC)

    def diagnose(deployed_hours_before: float) -> str:
        payload = _payload(
            [],
            signals={"memory_percent": 90, "restart_count": 2},
            memory_rise_started_at=rise,
            recent_change_events=[
                RecentChangeEvent(
                    change_id="DEP-1",
                    description="Deployment v3",
                    changed_at=rise - timedelta(hours=deployed_hours_before),
                )
            ],
        )
        return evaluate_hypotheses(payload)[0].code

    assert diagnose(0.5) == "MEMORY_LEAK_POST_DEPLOY"
    assert diagnose(48) == "MEMORY_PRESSURE"


def test_new_causes_get_targeted_recommendations() -> None:
    config = build_result(
        _payload(
            [
                _item("K8S", "k8s_event", "Pods restarting", reason="CrashLoopBackOff"),
                _item("LOG", "log", "Startup failed", lines=["configuration error: unknown field"]),
            ],
            recent_change_events=[
                RecentChangeEvent(
                    change_id="9f2c1ab",
                    description="Update configmap",
                    changed_at=datetime(2026, 10, 20, 9, 35, tzinfo=UTC),
                )
            ],
        )
    )
    dependency = build_result(
        _payload(
            [_item("LOG", "log", "Errors", lines=["connection refused"], dependency="payments-db")],
            signals={"error_rate": 0.25},
        )
    )

    assert [item.action for item in config.recommended_actions][
        0
    ] == "ROLLBACK_OR_REVERT_DEPLOYMENT"
    assert config.recommended_actions[0].parameters["deployment_id"] == "9f2c1ab"
    assert config.requires_additional_evidence is False
    assert dependency.recommended_actions[0].action == "INVESTIGATE_DEPENDENCY"
    assert dependency.recommended_actions[0].parameters["dependency"] == "payments-db"
    assert not {"ROLLBACK_OR_REVERT_DEPLOYMENT", "SCALE_OR_REBALANCE_RESOURCES"} & {
        item.action for item in dependency.recommended_actions
    }


def test_evidence_strength_rewards_variety_and_observed_provenance() -> None:
    result = build_result(
        _payload(
            [
                _item("K8S", "k8s_event", "Pods restarting", reason="CrashLoopBackOff"),
                _item("LOG", "log", "Startup failed", lines=["invalid configuration"]),
            ]
        )
    )
    derived_only = build_result(_payload([], signals={"cpu_percent": 95, "latency_ms": 900}))

    assert result.evidence_strength == round(0.6 * (2 / 3) + 0.4 * 1.0, 2)
    assert 0 < derived_only.evidence_strength < result.evidence_strength
    assert evidence_strength(result.hypotheses[0], []) == 0.0


def _change(text: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id="EVD-CHANGE",
        evidence_type="git_change",
        source="git",
        provenance="provided",
        observed_at=datetime(2026, 10, 20, 9, tzinfo=UTC),
        summary=text,
        value={"description": text},
    )


@pytest.mark.parametrize(
    ("text", "lowered"),
    [
        ("chore: lower memory limit to 256Mi to save cost", True),
        ("perf: align resources, limits.memory 1Gi -> 512Mi", True),
        ("set memory limit from 512Mi to 256Mi", True),
        ("raise limits.memory 512Mi -> 1Gi", False),
        ("memory limit 1Gi -> 1024Mi (same size)", False),
        ("fix memory leak in cache", False),
    ],
)
def test_memory_limit_lowered_compares_sizes(text: str, lowered: bool) -> None:
    assert memory_limit_lowered(_change(text)) is lowered


@pytest.mark.parametrize(
    "line",
    [
        "upstream connect error or disconnect/reset before headers. "
        "reset reason: connection failure",
        "rpc error: code = Unavailable desc = no healthy upstream",
        '"GET /v2/prices HTTP/1.1" 503 UF,URX upstream_reset_before_response_started',
    ],
)
def test_envoy_and_grpc_wording_is_a_dependency_failure(line: str) -> None:
    assert "upstream_unavailable" in line_signatures(line)
    assert "upstream_unavailable" not in line_signatures(
        "HTTP 503 from our own handler: maintenance mode"
    )
