"""Live test of Agent 1 on the team's fraud-detection API (about 20 minutes).

Needs: docker compose --profile observability --profile fraud up -d fraud-api prometheus cadvisor

  0. normal       healthy transactions                     -> service UP, no alert
  1. outage       docker stop fraud-api                    -> readiness HIGH, ALERT
  1b. repair      docker start + deployment.action.completed -> recovery check RECOVERED
  2. real bugs    old payment method / missing category    -> HTTP 500, SLO availability BREACH

The agent runs in its own process for the whole test; its key lines are shown live and the
full log is kept in experiments/runs/fraud/agent.log.

Usage: python experiments/checks/fraud_scenario.py
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/runs/fraud"
PY = sys.executable
COMPOSE = ["docker", "compose", "--profile", "observability", "--profile", "fraud"]
SHOW = re.compile(
    r"^\[\d\d:\d\d:\d\d\] cycle|Disponible|INCIDENT|DECISION :|=> EVENEMENT|-> (ANOMALY|BREACH)"
)
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def say(message: str) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] ===== {message}", flush=True)


def write_config() -> Path:
    config = yaml.safe_load((ROOT / "configs/monitoring.yaml").read_text(encoding="utf-8"))
    config.update(interval_seconds=30, service_timeout_seconds=25, logs_enabled=False)
    config["services"] = [s for s in config["services"] if s["service_id"] == "fraud-api"]
    config["forecast"]["enabled"] = False  # needs 30 min of history; RAM is short
    config["memory"]["path"] = str(OUT / "memory.sqlite")
    config["incident_store_path"] = str(OUT / "incidents.sqlite")
    config["events"]["path"] = str(OUT / "events.jsonl")
    path = OUT / "fraud.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def traffic(buggy_share: float) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            PY,
            "experiments/load/fraud_traffic.py",
            "--rps",
            "2",
            "--duration",
            "3600",
            "--buggy-share",
            str(buggy_share),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("events.jsonl", "memory.sqlite", "incidents.sqlite"):
        (OUT / old).unlink(missing_ok=True)
    config = write_config()
    log_path = OUT / "agent.log"
    healthy = traffic(0.0)
    with log_path.open("w", encoding="utf-8") as log:
        agent = subprocess.Popen(
            [
                PY,
                "-W",
                "ignore",
                "-m",
                "src.agents.monitoring",
                "--verbose",
                "--config",
                str(config),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=ENV,
        )
        shown = 0

        def pause(seconds: float) -> None:
            nonlocal shown
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                time.sleep(2)
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in lines[shown:]:
                    if SHOW.search(line):
                        print("   " + line.rstrip()[:150], flush=True)
                shown = len(lines)

        bug: subprocess.Popen[bytes] | None = None
        try:
            say("0. normal : transactions saines")
            pause(120)

            say("1. PANNE : arret de l'API de fraude (docker stop)")
            subprocess.run([*COMPOSE, "stop", "fraud-api"], cwd=ROOT, check=True)
            pause(180)

            say("1b. REPARATION : redemarrage, puis verification de la guerison")
            subprocess.run([*COMPOSE, "start", "fraud-api"], cwd=ROOT, check=True)
            verify = subprocess.run(
                [
                    PY,
                    "-W",
                    "ignore",
                    "-m",
                    "src.agents.monitoring.inbox",
                    "event",
                    "experiments/checks/fraud_action_completed.json",
                    "--window",
                    "120",
                    "--no-forecast",
                    "--config",
                    str(config),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=ENV,
                check=False,
            )
            (OUT / "recovery.log").write_text(verify.stdout + verify.stderr, encoding="utf-8")
            for line in verify.stdout.splitlines():
                if re.search(r'"(status|service_up|error_rate|latency_p95_ms)"', line):
                    print("   " + line.strip(), flush=True)
            pause(30)

            say("2. VRAIS BUGS : 50 % des transactions declenchent un bug de l'API (500)")
            healthy.terminate()
            bug = traffic(0.5)
            pause(420)
        finally:
            agent.terminate()
            agent.wait(timeout=30)
            healthy.terminate()
            if bug is not None:
                bug.terminate()
            subprocess.run(
                [*COMPOSE, "start", "fraud-api"], cwd=ROOT, check=False, capture_output=True
            )
    say(f"fin : journal complet {log_path}, evenements {OUT / 'events.jsonl'}")


if __name__ == "__main__":
    main()
