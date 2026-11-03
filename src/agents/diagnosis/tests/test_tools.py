import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.agents.diagnosis import tools
from src.agents.diagnosis.contracts import DiagnosisInput
from src.agents.diagnosis.investigation import investigate, plan_investigation
from src.agents.diagnosis.result_builder import build_result
from src.agents.diagnosis.tools import (
    GitChangesTool,
    KubernetesTool,
    LokiLogsTool,
    PrometheusMetricsTool,
    StaticEvidenceTool,
    ToolContext,
    ToolError,
    Toolbox,
    build_toolbox_from_env,
    load_static_toolbox,
    static_toolbox,
)


def _context(incident: str = "INC-T", deadline_minutes: float = 5) -> ToolContext:
    return ToolContext(
        incident_id=incident,
        service_id="payment-service",
        namespace="prod",
        deadline=datetime.now(UTC) + timedelta(minutes=deadline_minutes),
    )


class CountingTool:
    name = "counting"
    description = "Counts calls."
    operations = {"op": "Counts.", "boom": "Fails."}

    def __init__(self) -> None:
        self.calls = 0

    def run(self, operation, context):
        self.calls += 1
        if operation == "boom":
            raise RuntimeError("backend down")
        return StaticEvidenceTool("counting", {"op": [{"evidence_type": "metric", "summary": "ok"}]}).run(
            operation, context
        )


def test_toolbox_allowlist_cache_per_incident_and_circuit_breaker() -> None:
    tool = CountingTool()
    toolbox = Toolbox([tool], max_failures_per_tool=2)

    assert toolbox.call("counting", "drop_table", _context()).record.status == "skipped"
    assert toolbox.call("kubectl_delete", "op", _context()).record.status == "skipped"
    first = toolbox.call("counting", "op", _context("INC-A"))
    again = toolbox.call("counting", "op", _context("INC-A"))
    other = toolbox.call("counting", "op", _context("INC-B"))
    assert first.record.status == again.record.status == other.record.status == "succeeded"
    assert first.evidence[0].evidence_id.startswith("EVD-INC-A-COUNTING-OP")
    assert other.evidence[0].evidence_id.startswith("EVD-INC-B-")
    assert tool.calls == 2

    assert toolbox.call("counting", "boom", _context()).record.status == "failed"
    assert toolbox.call("counting", "boom", _context("INC-C")).record.status == "failed"
    opened = toolbox.call("counting", "op", _context("INC-D"))
    assert opened.record.status == "skipped" and "circuit open" in opened.record.error


def test_expired_deadline_skips_tool_calls() -> None:
    result = Toolbox([CountingTool()]).call("counting", "op", _context(deadline_minutes=-1))

    assert result.record.status == "skipped"
    assert "deadline" in result.record.error


def test_tool_definitions_are_strict_and_list_only_allowlisted_operations() -> None:
    [definition] = Toolbox([CountingTool()]).definitions()

    assert definition["strict"] is True
    assert definition["input_schema"]["additionalProperties"] is False
    assert definition["input_schema"]["properties"]["operation"]["enum"] == ["boom", "op"]


def test_kubernetes_tool_reads_pod_state_and_rollouts_with_fixed_readonly_commands() -> None:
    commands = []
    recent = (datetime.now(UTC) - timedelta(minutes=30)).replace(microsecond=0)

    def runner(argv, timeout):
        commands.append(list(argv))
        if argv[2] == "pods":
            return json.dumps({"items": [{"status": {"containerStatuses": [{
                "restartCount": 6,
                "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                "lastState": {"terminated": {"reason": "Error", "exitCode": 1}},
            }]}}]})
        return json.dumps({"items": [{
            "metadata": {
                "creationTimestamp": recent.isoformat().replace("+00:00", "Z"),
                "annotations": {"deployment.kubernetes.io/revision": "42"},
            },
            "spec": {"template": {"spec": {"containers": [{"image": "payment:v2.7"}]}}},
        }]})

    tool = KubernetesTool(runner=runner)
    [pods] = tool.run("pod_status", _context())
    [rollout] = tool.run("deployment_history", _context())

    assert pods.value["restart_count"] == 6
    assert "CrashLoopBackOff" in pods.value["reason"]
    assert rollout.value["change_id"] == "payment-service-rev42"
    assert rollout.value["changed_at"] == recent.isoformat()
    assert all(command[1] == "get" and "-o" in command for command in commands)
    assert ["-l", "app=payment-service"] == commands[0][5:7]


def test_unsafe_names_are_refused_before_any_command_runs() -> None:
    tool = KubernetesTool(runner=lambda argv, timeout: pytest.fail("command must not run"))
    context = ToolContext("INC", "svc; rm -rf /", "prod", datetime.now(UTC) + timedelta(minutes=1))

    with pytest.raises(ToolError, match="not a valid Kubernetes name"):
        tool.run("pod_status", context)


def test_git_tool_parses_commits_and_changed_files() -> None:
    committed = "2026-10-20T09:35:00+00:00"
    output = f"\x1e9f2c1ab0000000\x1f{committed}\x1fUpdate configmap\ndeploy/values.yaml\n"
    [item] = GitChangesTool(Path("."), runner=lambda argv, timeout: output).run("recent_changes", _context())

    assert item.value["change_id"] == "9f2c1ab00000"
    assert "deploy/values.yaml" in item.value["description"]
    assert item.value["changed_at"] == committed


def test_loki_reads_only_recent_errors_so_a_resolved_incident_does_not_pollute(monkeypatch) -> None:
    seen = {}

    def fake_get(base_url, path, params, timeout):
        seen.update(params)
        return {"data": {"result": []}}

    monkeypatch.setattr(tools, "_http_get_json", fake_get)
    LokiLogsTool("http://loki").run("error_logs", _context())  # context lookback: 1 hour
    window_minutes = (int(seen["end"]) - int(seen["start"])) / 1e9 / 60
    assert window_minutes == pytest.approx(15, abs=0.01)

    toolbox = tools.build_toolbox_from_env({"PIDS_LOKI_URL": "http://loki", "PIDS_LOKI_WINDOW_MINUTES": "40"})
    assert toolbox.tools[0]._window == timedelta(minutes=40)
    with pytest.raises(ToolError, match="PIDS_LOKI_WINDOW_MINUTES"):
        tools.build_toolbox_from_env({"PIDS_LOKI_URL": "http://loki", "PIDS_LOKI_WINDOW_MINUTES": "abc"})


def test_only_the_current_error_episode_is_kept() -> None:
    # Lab timeline: dependency errors at 14:26-14:27 (resolved), quiet, then pool errors from 14:33.
    def at(minute: int, second: int = 0) -> int:
        return int(datetime(2026, 11, 2, 14, minute, second, tzinfo=UTC).timestamp() * 1e9)

    entries = [
        (at(26, 40), "connect timed out to ledger-db:5432"),
        (at(27, 8), "dial tcp ledger-db:5432: connect: connection refused"),
        (at(33, 10), "connection pool exhausted (size=1)"),
        (at(34, 55), "connection pool exhausted (size=1)"),
        (at(35, 50), "connection pool exhausted (size=1)"),
    ]

    assert tools.current_error_episode(entries) == ["connection pool exhausted (size=1)"] * 3
    assert len(tools.current_error_episode(entries, gap=timedelta(minutes=10))) == 5
    assert tools.current_error_episode([]) == []


def test_prometheus_and_loki_tools_turn_responses_into_evidence(monkeypatch) -> None:
    def fake_get(base_url, path, params, timeout):
        if path == "/api/v1/query":
            # Real Prometheus rejects "sum(...) offset 1d" (HTTP 400): offset must follow a selector.
            assert not params["query"].endswith(") offset 1d"), params["query"]
            value = "700" if "[5m] offset 1d" in params["query"] else "1500"
            return {"status": "success", "data": {"result": [{"value": [0, value]}]}}
        return {"data": {"result": [{"values": [
            ["1", "dial tcp payments-db:5432: connect: connection refused"],
            ["2", "dial tcp payments-db:5432: connect: connection refused"],
        ]}]}}

    monkeypatch.setattr(tools, "_http_get_json", fake_get)
    [rate] = PrometheusMetricsTool("http://prom").run("request_rate", _context())
    [logs] = LokiLogsTool("http://loki").run("error_logs", _context())

    assert rate.value["rps"] == 1500 and rate.value["rps_baseline"] == 700
    assert logs.value["signatures"] == ["connection_refused"]
    assert logs.value["dependency"] == "payments-db"
    assert logs.value["line_count"] == 2


def test_live_toolbox_is_built_only_from_configured_settings(tmp_path: Path) -> None:
    assert build_toolbox_from_env({}) is None
    toolbox = build_toolbox_from_env(
        {"PIDS_PROMETHEUS_URL": "http://prom", "PIDS_DIAGNOSIS_GIT_REPO": str(tmp_path)}
    )
    assert toolbox is not None
    assert toolbox.available("prometheus", "memory_percent")
    assert toolbox.available("git", "recent_changes")
    assert not toolbox.available("kubernetes", "pod_status")


def test_recorded_toolbox_file_and_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"tools": {"loki": {"error_logs": [
        {"evidence_type": "log", "summary": "x", "value": {"lines": ["timeout"]}}]}}}), encoding="utf-8")

    result = load_static_toolbox(path).call("loki", "error_logs", _context())
    broken = static_toolbox({"loki": {"error_logs": [{"summary": "missing type"}]}}).call(
        "loki", "error_logs", _context()
    )

    assert result.evidence[0].provenance == "fixture"
    assert broken.record.status == "failed"


def _memory_input(**kwargs) -> DiagnosisInput:
    return DiagnosisInput(
        incident_id="INC-INV",
        service_id="payment-service",
        namespace="prod",
        environment="production",
        issue_summary="Memory rising.",
        signals={"memory_percent": 91, "restart_count": 2},
        memory_rise_started_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
        **kwargs,
    )


def test_plan_targets_the_incident_and_maps_replan_evidence() -> None:
    plan, unknown = plan_investigation(
        _memory_input(required_evidence=["heap_usage_breakdown", "gc_metrics", "thread_dump"])
    )

    assert ("prometheus", "memory_percent") in plan
    assert ("prometheus", "gc_pause") in plan
    assert ("prometheus", "cpu_percent") not in plan
    assert unknown == ["thread_dump"]


def test_collected_rollout_turns_memory_pressure_into_post_deploy_leak() -> None:
    toolbox = static_toolbox({"kubernetes": {"deployment_history": [{
        "evidence_type": "k8s_event",
        "summary": "Deployment revision 42 rolled out.",
        "value": {"change_id": "payment-service-rev42", "description": "Deployment revision 42",
                  "changed_at": "2026-10-20T11:40:00+00:00"},
    }]}})

    without_tools = build_result(_memory_input())
    with_tools = build_result(_memory_input(), toolbox=toolbox)
    investigation = investigate(_memory_input(), toolbox, datetime.now(UTC) + timedelta(minutes=1))

    assert without_tools.primary_root_cause == "MEMORY_PRESSURE"
    assert with_tools.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    assert with_tools.recommended_actions[0].parameters["deployment_id"] == "payment-service-rev42"
    assert [call.status for call in with_tools.tool_calls] == ["succeeded"]
    assert investigation.change_events[0].change_id == "payment-service-rev42"


def test_required_evidence_without_tools_is_reported() -> None:
    result = build_result(_memory_input(required_evidence=["gc_metrics"]))

    assert any("no evidence tools are configured" in item for item in result.limitations)
