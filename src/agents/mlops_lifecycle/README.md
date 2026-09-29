# Agent 6: MLOps lifecycle

Owner: Member 6. Agent 6 assesses model health locally, trains and compares a candidate on a
synthetic machine-failure dataset, stores the run and candidate in local MLflow, and requests
promotion review when a configured gate passes. Production data connectors and deployment are
separate work.

## Run local examples

The source-backed machine-failure example reads three separate JSON files: a saved baseline,
production-like prediction records and later confirmed outcomes. Its query selects model `v1`,
a one-hour window and a label cutoff. From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle --sources contracts/examples/agent6/local-sources/machine-sources.json
```

The CLI resolves file paths relative to `machine-sources.json`. The three local file adapters
validate records, select the requested version/window, match labels by observation ID, construct
histograms and run the health assessment. Its report recommends `RETRAIN_AND_EVALUATE` for this
synthetic dataset. File content SHA-256 digests identify the exact local inputs in the result;
they do not constitute a production evidence store. No raw observations are copied into the report.

You can also supply already-prepared health input:

```powershell
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle --input contracts/examples/agent6/model-health-input.json --policy configs/agent6-health-policy.json
```

The synthetic fixture returns `RETRAIN_AND_EVALUATE`: labeled F1 is approximately 0.667
against a baseline of 0.9, label coverage is 60%, and both distributions drift. No model is
trained or deployed by this command.

The record-based example builds the histograms and label alignment first:

```powershell
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle --records contracts/examples/agent6/observation-window.json
```

This older synthetic fixture includes numeric transaction amounts, categorical merchant values,
prediction scores and predicted classes. Confirmed labels are joined by observation ID.
Legacy `transaction_id` and `fraud_score` field names remain accepted when reading records;
new records use `observation_id` and `positive_class_score`.
A label confirmed after `as_of` is excluded; changing `as_of` permits a later assessment.
The reference bins and counts come from the versioned baseline embedded in this local fixture.
The record-based CLI path calls `MLOpsLifecycleAgent.assess_observation_window`.
The new `--sources` path calls `MLOpsLifecycleAgent.assess_model_window` through injected
baseline, prediction and label reader interfaces. The local HTTP API accepts only prepared
statistics; raw-record access is intentionally limited to local CLI examples at this stage.

Start the existing API with `python -m src.cli serve --reload` and open
`http://127.0.0.1:8000/docs`. Submit the prepared model-health-input.json example to `POST /api/v1/mlops/health/assess`.
The endpoint returns a full diagnostic report; invalid inputs and undersized histograms return
HTTP 422. The API currently uses the default policy. The CLI also accepts an explicit JSON policy.

## Run the complete local lifecycle

Install the optional ML dependencies in the same virtual environment. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle.demo
```

On macOS/Linux, use `.venv/bin/python` instead. The default command uses
`contracts/examples/agent6/local-sources/machine-sources.json` for a source-backed health check,
`configs/agent6-training-demo.json` for training gates, and 600 seeded synthetic historical
machine observations. Output is written to `artifacts/agent6-demo/`:

- `lifecycle-report.json`: health decision, split counts, champion/candidate metrics, final
  review recommendation, and limitations.
- `mlflow.db`: local SQLite tracking and model registry records.
- `mlruns/`: local model and run artifacts, including the policy and split manifest.

The flow is: read a saved baseline and monitored predictions/confirmed labels; compute PSI and
labeled F1; if quality has degraded, split historical examples 60/20/20 by time and machine group;
train the candidate only on the training split; compare it with a fixed legacy rule on the held-out
test split; record the run; register the candidate only if its F1 gain, recall and p95 latency pass
the policy. The validation split is reported separately. The legacy rule flags vibration of at
least 9 mm/s; the candidate uses temperature, vibration and machine type. Neither is a production
champion. A successful result says `REQUEST_PROMOTION_REVIEW`. It does not deploy the model.

For example, the seeded dataset produced a champion test F1 of 0.714 and candidate test F1 of
0.889 in local verification. Training labels are generated from a known synthetic relationship,
so these numbers demonstrate the pipeline, not model quality on real equipment. Latency varies
by machine. The health window and historical training set are separate synthetic fixtures.

The demo accepts `--sources`, `--training-policy`, `--output-dir`, `--rows` and `--seed`.
Use `--output-dir` for isolated experiments. Source JSON paths are resolved relative to the
source configuration file. Production adapters must replace these local file readers and
synthetic training data, and must supply the actual deployed champion for comparison. The
orchestrator and Agent 4 must enforce approval, deployment, canary checks and rollback.

## Inputs and provenance

`ModelHealthInput` in `src/contracts/mlops.py` contains model/version/reference identity,
a timezone-aware time window, histograms, evidence references and optional aligned labels.
The local window builder derives current histograms using the baseline's **fixed bin definitions**.
Numeric bin edges are lower-inclusive: an edge of 50 places 50 in the bin beginning at 50.
Categorical values outside listed categories enter an explicit other bin. Null or absent feature
values require an explicit missing bin; otherwise input construction fails. Production source
adapters must still apply the same feature preprocessing as the baseline. Bin names alone cannot
prove that upstream transformations match.

Each histogram needs at least 100 observations by default. Feature and prediction histograms
are evaluated independently. When prediction histograms and labeled predictions are supplied
together, their window sizes must match. Upstream adapters remain responsible for ensuring they
actually describe the same observations and window.

Labels must be 0, 1 or null. Position i in the labels and predictions arrays must identify
the same observation. Null means unavailable ground truth; it is excluded from F1 while
reducing coverage. The baseline F1 must come from comparable, representative evaluation data.
The service reports insufficient evidence if F1 is undefined, there are too few labeled
observations or positives, or coverage is too low. Sample counts cannot prove label representativeness.

For an orchestrator task, place the input under `task_context.signals.model_health` and
include every evidence reference in `task_context.evidence_refs`. The agent checks task identity,
deadline and this bounded evidence scope. Its `AgentResult.payload` contains the existing
`ModelHealthAssessed` event payload. No Kafka event is published or context entry persisted yet.

## Metrics and decision policy

PSI sums `(current - reference) * ln(current / reference)` across corresponding bin proportions.
The implementation adds epsilon to each proportion and renormalizes before calculating PSI
so empty bins remain finite. The aggregate score is the largest PSI across submitted distributions;
the report also includes every individual score.
See the [Arize PSI definition](https://arize.com/glossary/population-stability-index-psi/).
Changing bins or epsilon changes the measured score.

Binary F1 is `2 TP / (2 TP + FP + FN)` on labeled observations, following the
[scikit-learn F1 definition](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html).
An undefined denominator is reported as null. F1 drop is `max(0, baseline - current)`.

| Evidence | Recommendation |
| --- | --- |
| Missing or insufficient ground truth | COLLECT_LABELS |
| Sufficient labels and F1 drop meets/exceeds threshold | RETRAIN_AND_EVALUATE |
| Sufficient labels and F1 drop below threshold | HEALTHY |

Drift alone does not trigger retraining. A HEALTHY report can flag drift when labeled F1 remains
stable; continue monitoring that shift. Performance loss can recommend evaluation even without
detected drift.

Defaults in `configs/agent6-health-policy.json` are starting values, requiring calibration:
PSI threshold 0.2, F1 drop 0.05 (absolute), at least 30 labels, at least 5 positive labels,
coverage at least 0.5, and epsilon 0.000001. These are configurable policy values, not validated
business acceptance criteria. No decision confidence or statistical significance is claimed.

## Implementation map and next work

- `metrics.py`: PSI and binary confusion counts.
- `health.py`: policy gates, diagnostics and recommendations.
- `window_builder.py`: fixed-bin histograms, record filtering and delayed-label matching.
- `ports.py`: baseline, prediction, label, training and registry interfaces.
- `local_files.py`: development-only file readers with content-digest provenance.
- `service.py`: bounded task handling, health assessment and the local lifecycle gate.
- `__main__.py`: prepared-input, record-window or separate-source JSON CLI.
- `training.py`, `mlflow_registry.py` and `demo_data.py`: local training, tracking and synthetic data.
- `demo.py`: complete local lifecycle command.
- `src/contracts/mlops.py`, `src/contracts/mlops_observations.py` and `src/contracts/mlops_training.py`: validated public contracts.
- `tests/unit/test_mlops_health.py`, `tests/unit/test_mlops_window_builder.py`,
  `tests/unit/test_mlops_sources.py`, `tests/unit/test_mlops_lifecycle.py`,
  `tests/integration/test_mlops_api.py` and `tests/integration/test_mlops_lifecycle_demo.py`: behavior verification.

Next: trusted production record-source adapters and persisted baseline lookup; threshold calibration
with representative data; durable evidence/results and Kafka orchestration; actual deployed-champion
comparison; controlled promotion through the orchestrator and Agent 4.
