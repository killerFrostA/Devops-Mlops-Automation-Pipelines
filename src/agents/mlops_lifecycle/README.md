# Agent 6 · MLOps and model lifecycle

Owner: Member 6. DSO4, DSO5 / BO5. Distinguish feature/prediction drift from performance decline
with labels. Emit `mlops.model.health.assessed`; justify retraining, track candidates and compare
champion/challenger under approved metric, data-quality and latency constraints.

Use health, training and registry ports; offline stage slots are in `ml/`. Recommend promotion
through orchestrator → Agent 4 and verify canary with Monitoring. No dataset/model, training
pipeline or registry integration is active.
