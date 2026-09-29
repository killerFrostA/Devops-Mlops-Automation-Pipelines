# Agent 1 · Monitoring and anomaly detection

Owner: Member 1. DSO1 / BO1, BO2. Receive bounded tasks; gather Prometheus/Loki/Kubernetes/trace
evidence; detect anomalies; emit `monitoring.anomaly.detected`. Independently verify actions over an
observation window and emit `monitoring.recovery.verified`. Use the detector/verifier ports.

Start with thresholds/baselines, then compare statistical/ML techniques on held-out faults.
Do not write infrastructure or read full Context. Append results/evidence through ContextAppender.
`service.py` is currently a NOT_IMPLEMENTED implementation slot.
