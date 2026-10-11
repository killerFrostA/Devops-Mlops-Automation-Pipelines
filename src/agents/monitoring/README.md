# Agent 1 · Monitoring and anomaly detection

Owner: Member 1. DSO1 / BO1, BO2. Receive bounded tasks; gather Prometheus/Loki/Kubernetes/trace
evidence; detect anomalies; emit `monitoring.anomaly.detected`. Independently verify actions over an
observation window and emit `monitoring.recovery.verified`. Use the detector/verifier ports.

The agent reports observed facts (what, where, when, severity, confidence). It never states a
root cause: that is the Diagnosis agent's job.

## Pipeline (one cycle, every `interval_seconds`)

| Layer | Module | Role |
|---|---|---|
| Collect | `collectors/` | Prometheus (app metrics or cAdvisor container metrics), Loki logs |
| Detect | `detectors/slo.py` | SLO multi-window multi-burn-rate (availability, latency) |
| | `detectors/statistical.py` | memory growth (slope, R²), level shifts (median/MAD) |
| | `detectors/logs.py` | Drain3 templates: new patterns and surges; optional Gemini labels |
| | `detectors/forecast.py` | Amazon Chronos-Bolt forecast, zero-shot on CPU (optional extra) |
| Group | `grouping.py` | one incident per service + symptom category, cooldown |
| Reason | `reasoning.py` | dependency graph + noisy-OR: most probable origin service |
| Decide | `decision.py` | expected utility: alert / wait / collect more; business rules first |
| Remember | `memory.py`, `feedback.py` | SQLite alert history, operator verdicts, Beta precision |
| Emit | `events.py` | v1 `EventEnvelope` + `AnomalyDetected`, JSON Lines publisher |
| Verify | `recovery.py` | independent check after `deployment.action.completed` |
| Serve | `service.py` | `MonitoringAgent.handle`: `detect_anomaly`, `verify_recovery` tasks |

Any project can be watched without code changes: list its services in `configs/monitoring.yaml`
(`mode: app` with configurable metric names, or `mode: container` measured by cAdvisor).

## Run

```bash
pip install -e ".[monitoring]"                 # + ".[forecast]" for Chronos (CPU torch first)
docker compose --profile observability up -d target-app prometheus cadvisor
python -m src.agents.monitoring --verbose      # autonomous loop
python -m src.agents.monitoring.inbox detect payment-service
python -m src.agents.monitoring.inbox event experiments/checks/action_completed.json
python -m src.agents.monitoring.feedback list  # alerts; then: feedback vraie|fausse <n>
python experiments/checks/full_scenario.py     # live end-to-end test of every fault (~50 min)
```

## Not wired yet (platform side)

- Kafka: events are written to `experiments/runs/agent1-events.jsonl` through the
  `EventPublisher` port; plug the team's Kafka publisher in when it exists.
- ContextAppender: `src/platform/context` is not implemented; evidence ids are returned in
  `AgentResult.evidence_refs` meanwhile.
- Kubernetes events and traces: only Docker containers are covered on the local bench.
