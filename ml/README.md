# Fraud-detection model development

Fraud detection is the example target model, not an implemented product in this skeleton.
Keep offline training/evaluation code here and orchestration-facing lifecycle logic in Agent 6.
Do not train on downloaded/private data without an agreed dataset and data-access policy.

Implement dataset versioning, temporal/group-aware splits, leakage checks, preprocessing pipelines,
imbalance handling, calibrated outputs and reproducible run metadata. Compare candidate against
champion using approved business/quality measures; F1 improvement and latency thresholds in the
PDF are examples, not approved acceptance targets.

Register candidates in MLflow. Production promotion goes through the shared orchestrator/policy
and Agent 4 deployment path, with canary verification by Agents 1 and 6.
`pipeline.py` names the offline stages and fails explicitly until they are implemented.
