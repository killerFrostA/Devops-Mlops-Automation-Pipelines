"""End-to-end live test of Agent 1 on the test bench, phase by phase, with a final summary.

Needs the bench running (target-app, prometheus, cadvisor) with at least 30 min of normal
history. Each phase injects one fault on a different signal, runs the agent in a fresh process
and records when each detector first sees it:

  A. memory leak 30 MB/min -> alert + contract event; detect_anomaly task; recovery check
     before the repair (NOT_RECOVERED) and after it (RECOVERED)
  B. latency 800 ms        -> SLO latency, statistics, Chronos
  C. 50 % HTTP 500 errors  -> SLO availability
  D. CPU busy loop         -> statistics, Chronos

Usage: python experiments/checks/full_scenario.py [HH:MM]   (about 50 minutes)
       HH:MM: send normal traffic until that time before the first fault.
Outputs: experiments/runs/scenario/ (one log per step) and summary.md.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/runs/scenario"
APP = "http://127.0.0.1:8080"
PY = sys.executable
CYCLE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] cycle")
FINDING = re.compile(r"^\s+(SLO|Statistique|Chronos)\s+(\S+)\s+(\S+)?.*-> (ANOMALY|BREACH)")
# Lines worth showing live: incidents, decisions, events, task and recovery results.
LIVE = re.compile(
    r"INCIDENT|DECISION :|=> EVENEMENT|\"status\"|rationale_summary|-> (ANOMALY|BREACH)"
)


def post(path: str) -> str:
    request = urllib.request.Request(APP + path, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode()


def say(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def write_config() -> Path:
    config = yaml.safe_load((ROOT / "configs/monitoring.yaml").read_text(encoding="utf-8"))
    config.update(interval_seconds=30, service_timeout_seconds=25, logs_enabled=False)
    config["memory"]["path"] = str(OUT / "memory.sqlite")
    config["incident_store_path"] = str(OUT / "incidents.sqlite")
    config["events"]["path"] = str(OUT / "events.jsonl")
    path = OUT / "scenario.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run(name: str, args: list[str], config: Path) -> str:
    """Run an agent command, keep its output in <name>.log, return the output."""
    command = [PY, "-W", "ignore", "-m", *args, "--config", str(config)]
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    lines: list[str] = []
    with (
        (OUT / f"{name}.log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, **env},
        ) as process,
    ):
        assert process.stdout is not None
        for line in process.stdout:  # live: the full log on disk, key lines on screen
            log.write(line)
            log.flush()
            lines.append(line)
            if CYCLE.match(line) or LIVE.search(line):
                print("   " + line.rstrip()[:150], flush=True)
    return "".join(lines)


def first_detections(log: str) -> dict[str, str]:
    """{"Statistique memory_mb": "19:08:33", ...}: first cycle where each detector fired."""
    found: dict[str, str] = {}
    stamp = "?"
    for line in log.splitlines():
        if match := CYCLE.match(line):
            stamp = match.group(1)
        elif match := FINDING.match(line):
            layer, name, rule = match.group(1), match.group(2), match.group(3)
            key = f"{layer} {name}" + (f" {rule}" if layer == "SLO" else "")
            found.setdefault(key, stamp)
    return found


def delay(start: str, seen: str) -> str:
    seconds = (
        datetime.strptime(seen, "%H:%M:%S") - datetime.strptime(start, "%H:%M:%S")
    ).total_seconds()
    return f"{int(seconds // 60)} min {int(seconds % 60):02d} s"


def phase(
    name: str, inject: str, cycles: int, config: Path, expected: set[str]
) -> dict[str, object]:
    post("/chaos/reset")
    started = datetime.now().strftime("%H:%M:%S")
    say(f"{name} : injection {inject}")
    post(inject)
    log = run(name, ["src.agents.monitoring", "--cycles", str(cycles), "--verbose"], config)
    seen = first_detections(log)
    unexpected = sorted(k for k in seen if k.split()[1] not in expected)
    say(f"{name} : detections {seen}")
    return {
        "phase": name,
        "fault": inject,
        "injected_at": started,
        "detections": {k: (v, delay(started, v)) for k, v in seen.items()},
        "unexpected": unexpected,
        "alerts": log.count("!! DECISION : ALERTER"),
        "events": log.count("=> EVENEMENT"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("events.jsonl", "memory.sqlite", "incidents.sqlite"):
        (OUT / old).unlink(missing_ok=True)
    config = write_config()
    traffic = subprocess.Popen(
        [PY, "experiments/load/traffic.py", "--rps", "2", "--duration", "4000"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    results: list[dict[str, object]] = []
    recovery: dict[str, str] = {}
    try:
        if len(sys.argv) > 1:  # e.g. 19:05: normal traffic first, so baselines have history
            until = datetime.combine(
                datetime.now().date(), datetime.strptime(sys.argv[1], "%H:%M").time()
            )
            say(f"trafic normal jusqu'a {sys.argv[1]} (historique pour les references)")
            while datetime.now() < until:
                time.sleep(5)
        # A. memory leak: detection, alert event, task, recovery before/after the repair.
        results.append(phase("A-memoire", "/chaos/memory?mb_per_min=30", 14, config, {"memory_mb"}))
        say("A : tache detect_anomaly pendant la fuite")
        recovery["detect"] = run(
            "A-detect", ["src.agents.monitoring.inbox", "detect", "payment-service"], config
        )
        action = str(ROOT / "experiments/checks/action_completed.json")
        say("A : verification SANS reparation (fuite toujours active)")
        recovery["before"] = run(
            "A-verify-before",
            ["src.agents.monitoring.inbox", "event", action, "--window", "120", "--no-forecast"],
            config,
        )
        say("A : reparation (reset) puis verification")
        post("/chaos/reset")
        recovery["after"] = run(
            "A-verify-after",
            ["src.agents.monitoring.inbox", "event", action, "--window", "120", "--no-forecast"],
            config,
        )
        # B, C, D: comparison of the detection methods on other signals.
        results.append(
            phase(
                "B-latence",
                "/chaos/latency?ms=800",
                16,
                config,
                {"latency_p95_ms", "latency-500ms"},
            )
        )
        results.append(
            phase(
                "C-erreurs",
                "/chaos/errors?rate=0.5&kind=server_error",
                20,
                config,
                {"availability"},
            )
        )
        results.append(phase("D-cpu", "/chaos/cpu?seconds=600", 16, config, {"cpu_cores"}))
    finally:
        post("/chaos/reset")
        traffic.terminate()

    events_file = OUT / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
    statuses = {
        step: (re.search(r'"status": "(\w+)"', text) or [None, "absent"])[1]
        for step, text in recovery.items()
        if step != "detect"
    }
    lines = ["# Agent 1 - scenario complet", "", f"Date : {datetime.now():%Y-%m-%d %H:%M}", ""]
    lines += ["| Phase | Panne | Detecteur | Premiere detection | Delai |", "|---|---|---|---|---|"]
    for r in results:
        detections = r["detections"]
        assert isinstance(detections, dict)
        for key, (at, after) in sorted(detections.items(), key=lambda kv: kv[1][0]):
            lines.append(f"| {r['phase']} | {r['fault']} | {key} | {at} | {after} |")
        if r["unexpected"]:
            lines.append(f"| {r['phase']} | | **hors panne** : {r['unexpected']} | | |")
    lines += [
        "",
        f"Evenements envoyes (events.jsonl) : {len(events)} "
        f"({', '.join(sorted({e['topic'] for e in events}))})",
        "Tache detect_anomaly : "
        + ("anomalie" if "Anomalie sur" in recovery["detect"] else "aucune"),
        f"Verification avant reparation : {statuses.get('before')}",
        f"Verification apres reparation : {statuses.get('after')}",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
