"""Live test of the reasoning layer on a real call chain.

  --chain 2 (default): payment-service -> payments-db
  --chain 3          : checkout-api -> payment-service -> payments-db

Each trial injects one labelled fault (the true origin is known), lets it run, then records
which service the agent ranks first (reasoning) and which service it alerts on (decision).
The fault is then removed and the bench rests so the next trial starts clean.

Needs (once): docker compose --profile observability build target-app
              PAYMENT_DEPENDENCY_URL=http://payments-db:8080 docker compose \
                  --profile observability --profile chain up -d payments-db target-app \
                  prometheus cadvisor

              (--chain 3: add "--profile chain3" and "checkout-api" to that command)

Usage: python experiments/checks/reasoning_scenario.py [--chain 3]   (about 1 h 50)
Outputs: experiments/runs/reasoning/ or reasoning3/ (agent.log, summary.md)
"""

import argparse
import contextlib
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/runs/reasoning"
PY = sys.executable
COMPOSE = [
    "docker",
    "compose",
    "--profile",
    "observability",
    "--profile",
    "chain",
    "--profile",
    "chain3",
]
PAYMENT, DB, CHECKOUT = "http://127.0.0.1:8080", "http://127.0.0.1:8082", "http://127.0.0.1:8083"
# Rest > SLO short window (5 min) + reasoning window (5 min): the next trial starts clean.
FAULT_SECONDS, REST_SECONDS, WARMUP_SECONDS = 240, 720, 720
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
CYCLE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] cycle")
TOP = re.compile(r"^\s+1\. (\S+)\s+([0-9.]+)")
ALERT = re.compile(r"!! DECISION : ALERTER (\S+)")


def post(url: str, attempts: int = 3) -> None:
    """Fault injection call. A busy laptop once made it time out and abort the whole run."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=30):
                return
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts:
                raise
            time.sleep(5)


def compose(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], cwd=ROOT, check=True, capture_output=True)


def say(message: str) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] ===== {message}", flush=True)


# (name, true origin, inject, repair)
TRIALS: list[tuple[str, str, Callable[[], None], Callable[[], None]]] = [
    (
        "base lente",
        "payments-db",
        lambda: post(f"{DB}/chaos/latency?ms=800"),
        lambda: post(f"{DB}/chaos/reset"),
    ),
    (
        "paiement lent",
        "payment-service",
        lambda: post(f"{PAYMENT}/chaos/latency?ms=800"),
        lambda: post(f"{PAYMENT}/chaos/reset"),
    ),
    (
        "base en erreurs",
        "payments-db",
        lambda: post(f"{DB}/chaos/errors?rate=0.5"),
        lambda: post(f"{DB}/chaos/reset"),
    ),
    (
        "paiement en erreurs",
        "payment-service",
        lambda: post(f"{PAYMENT}/chaos/errors?rate=0.5"),
        lambda: post(f"{PAYMENT}/chaos/reset"),
    ),
    (
        "base arretee",
        "payments-db",
        lambda: compose("stop", "payments-db"),
        lambda: compose("start", "payments-db"),
    ),
    (
        "paiement arrete",
        "payment-service",
        lambda: compose("stop", "target-app"),
        lambda: compose("start", "target-app"),
    ),
]

# Chain of 3: the origin is at each level in turn; symptoms climb up to checkout-api.
TRIALS_3: list[tuple[str, str, Callable[[], None], Callable[[], None]]] = [
    TRIALS[0],  # base lente
    TRIALS[1],  # paiement lent
    (
        "checkout lent",
        "checkout-api",
        lambda: post(f"{CHECKOUT}/chaos/latency?ms=800"),
        lambda: post(f"{CHECKOUT}/chaos/reset"),
    ),
    TRIALS[2],  # base en erreurs
    TRIALS[3],  # paiement en erreurs
    TRIALS[4],  # base arretee
]


def write_config(chain: int) -> Path:
    config = yaml.safe_load((ROOT / "configs/monitoring.yaml").read_text(encoding="utf-8"))
    config.update(interval_seconds=30, service_timeout_seconds=25, logs_enabled=False)
    config["services"] = [
        {"service_id": "payment-service", "namespace": "local"},
        {"service_id": "payments-db", "namespace": "local"},
    ]
    if chain == 3:
        config["services"].insert(0, {"service_id": "checkout-api", "namespace": "local"})
    # No slow-burn: its 30-minute short window would carry a trial into the next one.
    config["slo"]["rules"] = [r for r in config["slo"]["rules"] if r["name"] != "slow-burn"]
    config["forecast"]["enabled"] = False
    config["memory"]["path"] = str(OUT / "memory.sqlite")
    config["incident_store_path"] = str(OUT / "incidents.sqlite")
    config["events"]["path"] = str(OUT / "events.jsonl")
    path = OUT / "reasoning.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def cycles_between(log: str, start: str, end: str) -> list[tuple[str, str | None, set[str]]]:
    """(time, first-ranked service, alerted services) for each cycle in [start, end)."""
    result: list[tuple[str, str | None, set[str]]] = []
    stamp, top, alerts = "", None, set()
    for line in [*log.splitlines(), "[99:99:99] cycle"]:
        if match := CYCLE.match(line):
            if stamp and start <= stamp < end:
                result.append((stamp, top, alerts))
            stamp, top, alerts = match.group(1), None, set()
        elif (match := TOP.match(line)) and top is None:
            top = match.group(1)
        elif match := ALERT.search(line):
            alerts.add(match.group(1))
    return result


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", type=int, choices=(2, 3), default=2)
    chain = parser.parse_args().chain
    trials = TRIALS_3 if chain == 3 else TRIALS
    entry = CHECKOUT if chain == 3 else PAYMENT  # clients call the front of the chain
    if chain == 3:
        OUT = ROOT / "experiments/runs/reasoning3"
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("events.jsonl", "memory.sqlite", "incidents.sqlite"):
        (OUT / old).unlink(missing_ok=True)
    config = write_config(chain)
    log_path = OUT / "agent.log"
    traffic = subprocess.Popen(
        [PY, "experiments/load/traffic.py", "--url", f"{entry}/pay", "--rps", "2"]
        + ["--duration", "7200"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    windows: list[tuple[str, str, str, str]] = []
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
        try:
            say(f"echauffement ({WARMUP_SECONDS // 60} min de trafic normal)")
            time.sleep(WARMUP_SECONDS)
            for index, (name, origin, inject, repair) in enumerate(trials, start=1):
                start = datetime.now().strftime("%H:%M:%S")
                say(f"essai {index}/{len(trials)} : {name} (vraie source : {origin})")
                inject()
                time.sleep(FAULT_SECONDS)
                end = datetime.now().strftime("%H:%M:%S")
                repair()
                windows.append((name, origin, start, end))
                say(f"essai {index} : reparation, puis {REST_SECONDS // 60} min de calme")
                time.sleep(REST_SECONDS)
        finally:
            agent.terminate()
            agent.wait(timeout=30)
            traffic.terminate()
            for _, _, _, repair in trials:  # best effort: leave the bench healthy
                with contextlib.suppress(Exception):
                    repair()

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = [
        "# Agent 1 - test reel du raisonnement (payment-service -> payments-db)",
        "",
        f"Date : {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| Essai | Vraie source | 1er classement | Classement majoritaire | Alerte sur | Bon ? |",
        "|---|---|---|---|---|---|",
    ]
    correct = 0
    for name, origin, start, end in windows:
        cycles = [c for c in cycles_between(log_text, start, end) if c[1] is not None]
        first = cycles[0][1] if cycles else None
        first_at = cycles[0][0] if cycles else "-"
        majority = Counter(c[1] for c in cycles).most_common(1)[0][0] if cycles else None
        alerted = sorted(set().union(*(c[2] for c in cycles))) if cycles else []
        ok = majority == origin
        correct += ok
        lines.append(
            f"| {name} | {origin} | {first} ({first_at}) | {majority} | "
            f"{', '.join(alerted) or '-'} | {'oui' if ok else 'NON'} |"
        )
    lines += ["", f"Precision (classement majoritaire = vraie source) : {correct}/{len(windows)}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
