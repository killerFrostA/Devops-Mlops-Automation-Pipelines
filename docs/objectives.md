# Objectives and measurement

These are project objectives from the supplied BO/DSO document. Targets must be agreed against a
measured baseline; the scaffold does not invent service levels or business benefits.

| BO | Operational value | Measures | Main components |
| --- | --- | --- | --- |
| BO1 | Reduce detection/recovery time | MTTD, MTTR, verified success | Agents 1, 2, 4; orchestrator |
| BO2 | Improve release reliability | Failed release and rollback rate | Agents 1, 3, 4 |
| BO3 | Increase safe automation | Autonomous resolution, HITL, false-action rate | Policy/HITL and action owners |
| BO4 | Improve resource efficiency | CPU/RAM, replicas, SLO compliance | Agents 5 and 1 |
| BO5 | Maintain model reliability | Drift delay, stable quality, candidate improvement | Agent 6 |
| BO6 | Explainable, traceable decisions | Audit completeness, evidence linkage | All agents and shared platform |

| DSO | Capability | Evaluation | Owner | BO linkage |
| --- | --- | --- | --- | --- |
| DSO1 | Telemetry anomaly detection | Precision, recall, F1, detection delay | Agent 1 | BO1 |
| DSO2 | Evidence-grounded RCA | Top-k correctness, diagnosis latency, evidence coverage | Agent 2 | BO1, BO2 |
| DSO3 | Resource forecasting | Forecast error, saving within SLO | Agent 5 | BO4 |
| DSO4 | Drift/degradation detection | False alarms, detection delay, precision/recall | Agent 6 | BO5 |
| DSO5 | Retraining/candidate evaluation | Candidate improvement, promotion correctness | Agent 6 | BO5 |
| DSO6 | Confidence/risk-aware decisions | Calibration, unsafe-action rate, escalation quality | All/shared | BO3, BO6 |

Kafka, LangGraph, gRPC and storage are architecture mechanisms, not business or data-science
objectives. Evaluate outcomes against a conventional/manual baseline using repeatable faults.
See `experiments/metrics.yaml` and `experiments/scenarios.yaml` for the measurement starting point.
