"""Drive the Agent 2 lab: a kind cluster where real faults are injected and diagnosed live.

    py -3 -m src.agents.diagnosis.lab.lab up                 # create cluster, build app, deploy
    py -3 -m src.agents.diagnosis.lab.lab inject leak        # leak | bad-config | dependency-down | pool
    py -3 -m src.agents.diagnosis.lab.lab run leak --wait    # Agent 1 stand-in event, then diagnosis
    py -3 -m src.agents.diagnosis.lab.lab reset              # back to the healthy release
    py -3 -m src.agents.diagnosis.lab.lab down               # delete the cluster

``event`` plays Agent 1: it reads the anomaly signals from the lab Prometheus and writes an
ANOMALY_DETECTED envelope. By default it gives no change context, so Agent 2 must find the
rollout itself with its live tools (kubectl, git, Prometheus, Loki).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[3]
OUT_DIR = LAB_DIR / "out"
STATE_FILE = OUT_DIR / "state.json"
APP_REPO = OUT_DIR / "payment-service-repo"
CLUSTER = "pids-lab"
KUBE_CONTEXT = f"kind-{CLUSTER}"
IMAGE = "pids-lab/app:1"
NAMESPACE = "prod"
SERVICE = "payment-service"
PROMETHEUS_URL = "http://127.0.0.1:9091"
LOKI_URL = "http://127.0.0.1:3100"
QDRANT_URL = "http://127.0.0.1:6333"
# Started by the project's compose file; its limit is set at run time so compose.yaml is untouched.
QDRANT_CONTAINER = "devops-mlops-automation-pipelines-qdrant-1"
QDRANT_MEMORY = "256m"
MIN_DOCKER_GIB = 3.5
SETTLE_MINUTES = 6.0
# Pulled once by Docker, then copied into the kind node (saves a second download on slow links).
PRELOAD_IMAGES = (
    "prom/prometheus:v3.5.0",
    "grafana/loki:3.5.0",
    "fluent/fluent-bit:4.0.3",
    "registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.16.0",
)
BASELINE_ENV = {
    "APP_VERSION": "1.3.0",
    "PAYMENT_TIMEOUT_MS": "500",
    "LEAK_KB_PER_REQUEST": "0",
    "DB_POOL_SIZE": "10",
    "DB_QUERY_MS": "20",
}


@dataclass(frozen=True)
class Fault:
    description: str
    env: dict[str, str]
    change_cause: str | None
    commit_file: str | None
    commit_content: str | None
    expected_root_cause: str
    expected_first_action: str
    min_minutes_after_inject: float
    # Anomaly condition checked by ``event --wait`` on the Agent 1 signals.
    wait_for: str


FAULTS = {
    "leak": Fault(
        "Release 1.4.0 keeps every response in memory: real growth up to the OOM kill.",
        {"APP_VERSION": "1.4.0", "LEAK_KB_PER_REQUEST": "24"},
        "release 1.4.0: in-memory response cache",
        "src/cache.py",
        "RESPONSE_CACHE = []  # never evicted\n",
        "MEMORY_LEAK_POST_DEPLOY",
        "ROLLBACK_OR_REVERT_DEPLOYMENT",
        6.0,
        "memory",
    ),
    "bad-config": Fault(
        "Release 1.4.1 sets PAYMENT_TIMEOUT_MS=5s: the new pod exits at start (CrashLoopBackOff).",
        {"APP_VERSION": "1.4.1", "PAYMENT_TIMEOUT_MS": "5s"},
        "config: PAYMENT_TIMEOUT_MS=5s",
        "config/values.yaml",
        "paymentTimeoutMs: 5s\n",
        "CONFIG_ERROR_CRASHLOOP",
        "ROLLBACK_OR_REVERT_DEPLOYMENT",
        1.0,
        "restarts",
    ),
    "dependency-down": Fault(
        "ledger-db is scaled to 0: payment-service gets connection refused (no release).",
        {},
        None,
        None,
        None,
        "DEPENDENCY_UNAVAILABLE",
        "INVESTIGATE_DEPENDENCY",
        1.0,
        "errors",
    ),
    "pool": Fault(
        "Release 1.4.2 shrinks the DB pool to 1 with slow queries: pool exhaustion under load.",
        {"APP_VERSION": "1.4.2", "DB_POOL_SIZE": "1", "DB_QUERY_MS": "400"},
        "config: DB pool size 1",
        "config/values.yaml",
        "dbPoolSize: 1\ndbQueryMs: 400\n",
        "CONNECTION_POOL_EXHAUSTION",
        "TUNE_CONNECTION_POOL",
        1.0,
        "errors",
    ),
}

_PODS = f'namespace="{NAMESPACE}",pod=~"{SERVICE}-.*"'
_SERVICE = f'namespace="{NAMESPACE}",service="{SERVICE}"'
SIGNAL_QUERIES = {
    "memory_pct": (
        f"100 * max(container_memory_working_set_bytes{{{_PODS}}}) / "
        f'max(kube_pod_container_resource_limits{{{_PODS},resource="memory"}})'
    ),
    "memory_slope_mb_min": (
        f'deriv(max(container_memory_working_set_bytes{{{_PODS},container="{SERVICE}"}})[10m:30s]) '
        "* 60 / 1048576"
    ),
    "latency_p95_ms": (
        "1000 * histogram_quantile(0.95, sum by (le) "
        f"(rate(http_request_duration_seconds_bucket{{{_SERVICE}}}[5m])))"
    ),
    "error_rate": (
        f'sum(rate(http_requests_total{{{_SERVICE},status=~"5.."}}[2m])) / '
        f"sum(rate(http_requests_total{{{_SERVICE}}}[2m]))"
    ),
    "pod_restarts": f"max(increase(kube_pod_container_status_restarts_total{{{_PODS}}}[30m]))",
}


class LabError(RuntimeError):
    pass


def _run(argv: list[str], *, check: bool = True, capture: bool = False, cwd: Path | None = None) -> str:
    print("$ " + " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False, text=True, capture_output=capture, cwd=cwd)  # noqa: S603
    if check and completed.returncode != 0:
        detail = (completed.stderr or "").strip() if capture else ""
        raise LabError(f"{argv[0]} exited with {completed.returncode} {detail}".strip())
    return completed.stdout or ""


def _kubectl(*args: str, capture: bool = False, check: bool = True) -> str:
    return _run(["kubectl", "--context", KUBE_CONTEXT, *args], capture=capture, check=check)


def _git(*args: str) -> str:
    return _run(
        ["git", "-C", str(APP_REPO), "-c", "user.name=pids-lab", "-c", "user.email=lab@localhost", *args],
        capture=True,
    )


def _load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _init_app_repo() -> None:
    if (APP_REPO / ".git").exists():
        return
    APP_REPO.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q", str(APP_REPO)], capture=True)
    (APP_REPO / "config").mkdir(exist_ok=True)
    (APP_REPO / "config" / "values.yaml").write_text("paymentTimeoutMs: 500\ndbPoolSize: 10\n", encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-q", "-m", "release 1.3.0 (baseline)")


def _commit_change(fault: Fault) -> None:
    if fault.commit_file is None or fault.commit_content is None:
        return
    path = APP_REPO / fault.commit_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fault.commit_content, encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-q", "-m", fault.change_cause or fault.description)


# Commands --------------------------------------------------------------------------------


def _docker_gib() -> float:
    memory = _run(["docker", "info", "--format", "{{.MemTotal}}"], capture=True).strip()
    return int(memory) / 1024**3 if memory.isdigit() else 0.0


def _running_containers() -> list[str]:
    return _run(["docker", "ps", "--format", "{{.Names}}"], capture=True).split()


def cmd_check(_: argparse.Namespace) -> None:
    for tool in ("docker", "kind", "kubectl", "git"):
        print(f"{tool:8s} {shutil.which(tool) or 'MISSING'}")
    gib = _docker_gib()
    print(f"Docker memory: {gib:.1f} GiB" + (f"  (below {MIN_DOCKER_GIB} GiB: the lab may not fit)" if gib < MIN_DOCKER_GIB else ""))
    clusters = _run(["kind", "get", "clusters"], capture=True).split()
    print(f"kind clusters: {clusters or 'none'}")
    print(f"running containers: {_running_containers() or 'none'}")


def cmd_qdrant(args: argparse.Namespace) -> None:
    """Start or stop the compose Qdrant container, capped at QDRANT_MEMORY."""
    if args.action == "stop":
        _run(["docker", "stop", QDRANT_CONTAINER], check=False)
        return
    if QDRANT_CONTAINER not in _run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture=True).split():
        _run(["docker", "compose", "--profile", "intelligence", "create", "qdrant"], cwd=REPO_ROOT)
    _run(["docker", "update", "--memory", QDRANT_MEMORY, "--memory-swap", QDRANT_MEMORY, QDRANT_CONTAINER])
    _run(["docker", "start", QDRANT_CONTAINER])


def _load_image(image: str) -> None:
    """Copy one image into the kind node, amd64 only.

    ``kind load docker-image`` imports with --all-platforms, which fails on Docker Desktop's
    containerd image store when only the local platform is present ("content digest not found").
    """
    node = f"{CLUSTER}-control-plane"
    present = _run(["docker", "exec", node, "ctr", "-n", "k8s.io", "images", "ls", "-q"], capture=True).split()
    if any(name.removeprefix("docker.io/library/").removeprefix("docker.io/") == image for name in present):
        print(f"{image}: already on {node}")
        return
    print(f"$ docker save --platform linux/amd64 {image} | docker exec -i {node} ctr images import", flush=True)
    save = subprocess.Popen(["docker", "save", "--platform", "linux/amd64", image], stdout=subprocess.PIPE)  # noqa: S603
    load = subprocess.run(  # noqa: S603
        ["docker", "exec", "-i", node, "ctr", "-n", "k8s.io", "images", "import",
         "--platform", "linux/amd64", "--digests", "--snapshotter=overlayfs", "-"],
        stdin=save.stdout, capture_output=True, text=True, check=False,
    )
    save.stdout.close()
    if save.wait() != 0 or load.returncode != 0:
        raise LabError(f"cannot load {image} into {node}: {load.stderr.strip()[:300]}")


def cmd_up(_: argparse.Namespace) -> None:
    others = [name for name in _run(["kind", "get", "clusters"], capture=True).split() if name != CLUSTER]
    if others:
        raise LabError(f"other kind clusters use memory: {others}; delete them first")
    if _docker_gib() < MIN_DOCKER_GIB:
        print(f"Warning: Docker has {_docker_gib():.1f} GiB; the lab needs about {MIN_DOCKER_GIB} GiB.")
    if CLUSTER not in _run(["kind", "get", "clusters"], capture=True).split():
        _run(["kind", "create", "cluster", "--config", str(LAB_DIR / "kind-cluster.yaml")])
    _run(["docker", "build", "-t", IMAGE, str(LAB_DIR / "app")])
    for image in PRELOAD_IMAGES:
        if not _run(["docker", "images", "-q", image], capture=True).strip():
            _run(["docker", "pull", image])
    for image in (IMAGE, *PRELOAD_IMAGES):
        _load_image(image)
    _kubectl("apply", "-f", str(LAB_DIR / "k8s" / "monitoring.yaml"))
    _kubectl("apply", "-f", str(LAB_DIR / "k8s" / "workload.yaml"))
    for namespace, name in (
        ("monitoring", "prometheus"), ("monitoring", "kube-state-metrics"), ("monitoring", "loki"),
        (NAMESPACE, "ledger-db"), (NAMESPACE, SERVICE), (NAMESPACE, "loadgen"),
    ):
        _kubectl("-n", namespace, "rollout", "status", f"deployment/{name}", "--timeout=300s")
    _kubectl("-n", "monitoring", "rollout", "status", "daemonset/fluent-bit", "--timeout=300s")
    _init_app_repo()
    _save_state({"fault": None, "injected_at": None})
    print(f"\nLab ready. Prometheus {PROMETHEUS_URL}, Loki {LOKI_URL}. Next: inject a fault.")


def cmd_status(_: argparse.Namespace) -> None:
    _kubectl("get", "pods", "-A", "-o", "wide", check=False)
    for name, url in (("Prometheus", f"{PROMETHEUS_URL}/-/ready"), ("Loki", f"{LOKI_URL}/ready")):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                print(f"{name}: HTTP {response.status}")
        except OSError as exc:
            print(f"{name}: not reachable ({exc})")
    print(f"State: {_load_state()}")
    signals = read_signals()
    print("Signals: " + json.dumps(signals))


def cmd_inject(args: argparse.Namespace) -> None:
    fault = FAULTS[args.fault]
    state = _load_state()
    if state.get("fault"):
        raise LabError(f"fault {state['fault']} is active; run reset first")
    if state.get("reset_at"):
        # Series of the replaced pods stay visible ~5 min (staleness) and in the 5m rate windows;
        # injecting earlier mixes the previous fault into the new incident's signals.
        remaining = SETTLE_MINUTES * 60 - (datetime.now(UTC) - datetime.fromisoformat(state["reset_at"])).total_seconds()
        if remaining > 0:
            print(f"Waiting {remaining / 60:.1f} min for the previous fault's signals to settle...", flush=True)
            time.sleep(remaining)
    _init_app_repo()
    print(fault.description)
    if args.fault == "dependency-down":
        _kubectl("-n", NAMESPACE, "scale", "deployment/ledger-db", "--replicas=0")
    else:
        _commit_change(fault)
        _kubectl("-n", NAMESPACE, "annotate", f"deployment/{SERVICE}",
                 f"kubernetes.io/change-cause={fault.change_cause}", "--overwrite")
        _kubectl("-n", NAMESPACE, "set", "env", f"deployment/{SERVICE}",
                 *(f"{key}={value}" for key, value in fault.env.items()))
    _save_state({"fault": args.fault, "injected_at": datetime.now(UTC).isoformat()})
    print(f"Injected {args.fault}. Wait at least {fault.min_minutes_after_inject:g} min, then: run {args.fault} --wait")


def cmd_reset(_: argparse.Namespace) -> None:
    state = _load_state()
    _kubectl("-n", NAMESPACE, "scale", "deployment/ledger-db", "--replicas=1")
    if state.get("fault") not in (None, "dependency-down"):
        _kubectl("-n", NAMESPACE, "annotate", f"deployment/{SERVICE}",
                 "kubernetes.io/change-cause=rollback to 1.3.0", "--overwrite")
        _kubectl("-n", NAMESPACE, "set", "env", f"deployment/{SERVICE}",
                 *(f"{key}={value}" for key, value in BASELINE_ENV.items()))
        _git("revert", "--no-edit", "HEAD")
    _kubectl("-n", NAMESPACE, "rollout", "status", f"deployment/{SERVICE}", "--timeout=180s", check=False)
    _save_state({"fault": None, "injected_at": None, "reset_at": datetime.now(UTC).isoformat()})
    print(f"Back to the healthy release. The next inject waits {SETTLE_MINUTES:g} min for the signals to settle.")


def cmd_down(_: argparse.Namespace) -> None:
    _run(["kind", "delete", "cluster", "--name", CLUSTER])
    _save_state({"fault": None, "injected_at": None})


# Agent 1 stand-in ------------------------------------------------------------------------


def prometheus_value(query: str, base_url: str = PROMETHEUS_URL) -> float | None:
    url = f"{base_url}/api/v1/query?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.load(response)
    result = data.get("data", {}).get("result", [])
    if not result:
        return None
    value = float(result[0]["value"][1])
    return None if value != value else value  # NaN when there is no traffic.


def read_signals(base_url: str = PROMETHEUS_URL) -> dict[str, float]:
    signals = {}
    for key, query in SIGNAL_QUERIES.items():
        try:
            value = prometheus_value(query, base_url)
        except OSError as exc:
            raise LabError(f"Prometheus not reachable at {base_url}: {exc}") from exc
        if value is not None:
            signals[key] = round(value, 4 if key == "error_rate" else 1)
    return signals


def anomaly_present(fault: Fault, signals: dict[str, float]) -> bool:
    if fault.wait_for == "memory":
        return signals.get("memory_pct", 0) >= 75 or signals.get("pod_restarts", 0) >= 1
    if fault.wait_for == "restarts":
        return signals.get("pod_restarts", 0) >= 2
    return signals.get("error_rate", 0) >= 0.05


def latest_rollout() -> tuple[str, datetime] | None:
    """Newest payment-service ReplicaSet revision and creation time, from kubectl."""
    output = _kubectl("-n", NAMESPACE, "get", "replicasets", "-l", f"app={SERVICE}", "-o", "json", capture=True)
    best = None
    for item in json.loads(output).get("items", []):
        metadata = item.get("metadata", {})
        revision = metadata.get("annotations", {}).get("deployment.kubernetes.io/revision")
        created = metadata.get("creationTimestamp")
        if revision and created:
            candidate = (int(revision), datetime.fromisoformat(created.replace("Z", "+00:00")))
            best = max(best, candidate) if best else candidate
    return (f"{SERVICE}-rev{best[0]}", best[1]) if best else None


def build_agent1_event(
    scenario: str,
    signals: dict[str, float],
    detected_at: datetime,
    change: tuple[str, datetime] | None = None,
) -> dict[str, Any]:
    """ANOMALY_DETECTED envelope in Agent 1's format (see tests/fixtures)."""
    stamp = detected_at.strftime("%Y%m%dT%H%M%SZ")
    incident_id = f"INC-LAB-{scenario.upper()}-{stamp}"
    payload_signals: dict[str, Any] = {**signals, "detected_at": detected_at.isoformat()}
    if change is not None:
        payload_signals["change_context"] = {
            "deployment_id": change[0],
            "minutes_since_deployment": round((detected_at - change[1]).total_seconds() / 60, 1),
        }
    severity_high = signals.get("memory_pct", 0) >= 85 or signals.get("error_rate", 0) >= 0.1
    return {
        "message_id": f"MSG-LAB-{scenario}-{stamp}",
        "schema_version": "1.0",
        "correlation_id": incident_id,
        "trace_id": f"TRACE-LAB-{stamp}",
        "source": "monitoring-agent",
        "target": "orchestrator",
        "topic": "monitoring.anomaly.detected",
        "message_type": "ANOMALY_DETECTED",
        "priority": "P1" if severity_high else "P2",
        "severity": "HIGH" if severity_high else "MEDIUM",
        "timestamp": detected_at.isoformat(),
        "deadline": (detected_at + timedelta(minutes=15)).isoformat(),
        "environment": "production",
        "service": {"service_id": SERVICE, "namespace": NAMESPACE, "version": "unknown"},
        "evidence_refs": [],
        "confidence": 0.9,
        "requires_ack": True,
        "idempotency_key": f"{incident_id}:ANOMALY_DETECTED:v1",
        "payload": {
            "anomaly_id": f"ANM-LAB-{stamp}",
            "detector": "pids-lab-threshold",
            "anomaly_score": 0.9,
            "signals": payload_signals,
            "verification_required_after_action": True,
        },
    }


def cmd_event(args: argparse.Namespace) -> Path:
    fault = FAULTS[args.fault]
    state = _load_state()
    if state.get("fault") != args.fault:
        raise LabError(f"fault {args.fault} is not injected (active: {state.get('fault')})")
    injected_at = datetime.fromisoformat(state["injected_at"])
    deadline = time.monotonic() + args.timeout_minutes * 60
    while True:
        signals = read_signals()
        elapsed = (datetime.now(UTC) - injected_at).total_seconds() / 60
        ready = elapsed >= fault.min_minutes_after_inject and anomaly_present(fault, signals)
        print(f"[{elapsed:5.1f} min] {json.dumps(signals)}" + ("  -> anomaly" if ready else ""), flush=True)
        if ready or not args.wait:
            break
        if time.monotonic() > deadline:
            raise LabError(f"no anomaly after {args.timeout_minutes} min")
        time.sleep(30)
    change = latest_rollout() if args.with_change_context else None
    event = build_agent1_event(args.fault, signals, datetime.now(UTC), change)
    directory = OUT_DIR / "incidents"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{event['correlation_id']}.agent1.json"
    path.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
    truth = {
        "fault": args.fault,
        "description": fault.description,
        "injected_at": state["injected_at"],
        "expected_root_cause": fault.expected_root_cause,
        "expected_first_action": fault.expected_first_action,
    }
    path.with_suffix("").with_suffix(".truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    print(f"Agent 1 event: {path}")
    return path


def live_tool_env() -> dict[str, str]:
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        raise LabError("kubectl not found in PATH")
    return {
        "PIDS_PROMETHEUS_URL": PROMETHEUS_URL,
        "PIDS_LOKI_URL": LOKI_URL,
        "PIDS_DIAGNOSIS_KUBECTL": kubectl,
        "PIDS_DIAGNOSIS_KUBE_CONTEXT": KUBE_CONTEXT,
        "PIDS_DIAGNOSIS_GIT_REPO": str(APP_REPO),
    }


def diagnose(event_path: Path, *, qdrant: bool, llm: str | None) -> None:
    argv = [sys.executable, "-m", "src.agents.diagnosis.agent1_runner", str(event_path), "--live-tools",
            "--output-dir", str(event_path.parent / event_path.name.split(".")[0])]
    if qdrant:
        argv += ["--qdrant-url", QDRANT_URL]
    if llm:
        argv += ["--llm", llm]
    print("$ " + " ".join(argv), flush=True)
    completed = subprocess.run(argv, cwd=REPO_ROOT, env={**os.environ, **live_tool_env()}, check=False)  # noqa: S603
    if completed.returncode != 0:
        raise LabError(f"diagnosis exited with {completed.returncode}")


SCENARIO_DIR = LAB_DIR.parent / "evaluation" / "scenarios"
LAB_KNOWLEDGE = LAB_DIR.parent / "knowledge" / "incidents_lab.json"
FAULT_TAGS = {
    "leak": ["memory", "leak", "deployment", "oom", "restart"],
    "bad-config": ["crashloop", "config", "deployment", "restart", "env"],
    "dependency-down": ["dependency", "database", "connection", "refused"],
    "pool": ["dependency", "pool", "connection", "exhausted", "config"],
}
_RECORDED_KEYS = ("evidence_type", "source", "summary", "value", "observed_at")


def record_tool_responses(incident_id: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Call every live read-only tool operation once and keep the answers for offline replay."""
    from src.agents.diagnosis.tools import ToolContext, build_toolbox_from_env

    toolbox = build_toolbox_from_env(live_tool_env())
    if toolbox is None:
        raise LabError("no live tool configured")
    context = ToolContext(incident_id, SERVICE, NAMESPACE, deadline=datetime.now(UTC) + timedelta(minutes=3))
    recorded: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for tool in toolbox.tools:
        for operation in tool.operations:
            result = toolbox.call(tool.name, operation, context)
            if result.record.status != "succeeded":
                print(f"  {tool.name}.{operation}: {result.record.status} {result.record.error or ''}")
            items = [
                {key: value for key, value in item.model_dump(mode="json").items() if key in _RECORDED_KEYS}
                for item in result.evidence
            ]
            if items:
                recorded.setdefault(tool.name, {})[operation] = items
    return recorded


def scenario_from_lab(
    scenario_id: str,
    event: dict[str, Any],
    truth: dict[str, Any],
    tool_responses: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Evaluation scenario from a real lab incident: Agent 1 event, recorded tools, ground truth."""
    return {
        "scenario_id": scenario_id,
        "title": f"Lab (real run): {truth['description']}",
        "architecture_ref": "18.1 Live lab incident (kind), recorded tool responses",
        "event": {
            "incident_id": event["correlation_id"],
            "service_id": event["service"]["service_id"],
            "namespace": event["service"]["namespace"],
            "environment": event["environment"],
            "severity": event["severity"],
            "priority": event["priority"],
            "detected_at": event["payload"]["signals"].get("detected_at", event["timestamp"]),
            "signals": event["payload"]["signals"],
        },
        "tool_responses": tool_responses,
        "expected": {
            "root_cause": truth["expected_root_cause"],
            "first_action": truth["expected_first_action"],
        },
    }


def rag_document_from_lab(event: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_id": f"LAB-{event['correlation_id'].removeprefix('INC-LAB-')}",
        "summary": f"{truth['description']} Lab incident on {SERVICE}, injected {truth['injected_at'][:16]}Z.",
        "tags": FAULT_TAGS.get(truth["fault"], []),
        "root_cause": truth["expected_root_cause"],
        "origin": "lab_observation",
    }


def _next_scenario_id(fault: str, directory: Path = SCENARIO_DIR) -> str:
    numbers = [int(path.name[1:3]) for path in directory.glob("S[0-9][0-9]-*.json")]
    return f"S{max(numbers, default=0) + 1:02d}-lab-{fault}"


def capture(event_path: Path, *, rag: bool = True) -> Path:
    event = json.loads(event_path.read_text(encoding="utf-8"))
    truth_path = event_path.with_name(event_path.name.replace(".agent1.json", ".truth.json"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    print("Recording live tool responses...")
    responses = record_tool_responses(event["correlation_id"])
    scenario_id = _next_scenario_id(truth["fault"])
    scenario_path = SCENARIO_DIR / f"{scenario_id}.json"
    scenario = scenario_from_lab(scenario_id, event, truth, responses)
    scenario_path.write_text(json.dumps(scenario, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scenario: {scenario_path}")
    if rag:
        try:
            corpus = json.loads(LAB_KNOWLEDGE.read_text(encoding="utf-8"))
        except OSError:
            corpus = {"format_version": 1, "documents": []}
        document = rag_document_from_lab(event, truth)
        corpus["documents"] = [
            item for item in corpus["documents"] if item["reference_id"] != document["reference_id"]
        ] + [document]
        LAB_KNOWLEDGE.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"RAG document {document['reference_id']} added to {LAB_KNOWLEDGE.name}; "
              "re-index Qdrant with: py -3 -m src.agents.diagnosis.knowledge_index --qdrant-url "
              f"{QDRANT_URL}")
    return scenario_path


def cmd_run(args: argparse.Namespace) -> None:
    event_path = cmd_event(args)
    if args.capture:
        # Record before diagnosing so the scenario holds the state Agent 2 saw.
        capture(event_path, rag=True)
    diagnose(event_path, qdrant=args.qdrant, llm=args.llm)


def cmd_capture(args: argparse.Namespace) -> None:
    capture(args.event, rag=not args.no_rag)


def cmd_diagnose(args: argparse.Namespace) -> None:
    diagnose(args.event, qdrant=args.qdrant, llm=args.llm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 2 live lab on kind.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check tools and Docker memory.").set_defaults(func=cmd_check)
    commands.add_parser("up", help="Create the kind cluster and deploy the lab.").set_defaults(func=cmd_up)
    commands.add_parser("status", help="Pods, endpoints and current signals.").set_defaults(func=cmd_status)
    commands.add_parser("reset", help="Remove the active fault.").set_defaults(func=cmd_reset)
    commands.add_parser("down", help="Delete the kind cluster.").set_defaults(func=cmd_down)
    qdrant = commands.add_parser("qdrant", help=f"Start (capped at {QDRANT_MEMORY}) or stop the compose Qdrant.")
    qdrant.add_argument("action", choices=("start", "stop"))
    qdrant.set_defaults(func=cmd_qdrant)
    inject = commands.add_parser("inject", help="Inject a fault.")
    inject.add_argument("fault", choices=sorted(FAULTS))
    inject.set_defaults(func=cmd_inject)
    for name, func, help_text in (
        ("event", cmd_event, "Write the Agent 1 event for the active fault."),
        ("run", cmd_run, "Write the Agent 1 event, then diagnose it with live tools."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("fault", choices=sorted(FAULTS))
        command.add_argument("--wait", action="store_true", help="Poll until the anomaly is visible.")
        command.add_argument("--timeout-minutes", type=float, default=20)
        command.add_argument("--with-change-context", action="store_true",
                             help="Let Agent 1 include the latest rollout (default: Agent 2 finds it).")
        if name == "run":
            command.add_argument("--qdrant", action="store_true", help=f"Use the Qdrant server at {QDRANT_URL}.")
            command.add_argument("--llm", nargs="?", const="openai", choices=("openai", "anthropic"))
            command.add_argument("--capture", action="store_true",
                                 help="Also save the incident as an evaluation scenario and a RAG document.")
        command.set_defaults(func=func)
    capture_command = commands.add_parser("capture", help="Save a lab event as a scenario and a RAG document.")
    capture_command.add_argument("event", type=Path)
    capture_command.add_argument("--no-rag", action="store_true")
    capture_command.set_defaults(func=cmd_capture)
    rerun = commands.add_parser("diagnose", help="Diagnose an existing lab event file.")
    rerun.add_argument("event", type=Path)
    rerun.add_argument("--qdrant", action="store_true")
    rerun.add_argument("--llm", nargs="?", const="openai", choices=("openai", "anthropic"))
    rerun.set_defaults(func=cmd_diagnose)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except LabError as exc:
        print(f"Lab error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
