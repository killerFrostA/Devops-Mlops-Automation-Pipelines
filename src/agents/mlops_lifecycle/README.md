# Agent 6: MLOps lifecycle

Owner: Member 6. The first implementation assesses model health locally. It measures feature
and prediction distribution drift, evaluates binary F1 on available ground truth, and returns
a typed recommendation. Training, registry operations, persistence and promotion are pending.

## Run the example

From the repository root in PowerShell:

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

Its 100 synthetic prediction records include numeric transaction amounts, categorical merchant
values, fraud scores and predicted classes. Confirmed labels are joined by transaction ID.
A label confirmed after `as_of` is excluded; changing `as_of` permits a later assessment.
The reference bins and counts come from the versioned baseline embedded in this local fixture.
The record-based CLI path calls `MLOpsLifecycleAgent.assess_observation_window`.
Only the prepared-statistics path is currently exposed by the local HTTP API.

Start the existing API with `python -m src.cli serve --reload` and open
`http://127.0.0.1:8000/docs`. Submit the prepared model-health-input.json example to `POST /api/v1/mlops/health/assess`.
The endpoint returns a full diagnostic report; invalid inputs and undersized histograms return
HTTP 422. The API currently uses the default policy. The CLI also accepts an explicit JSON policy.

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
- `service.py`: bounded task handling, record-window entry point and injected assessor port.
- `__main__.py`: prepared-input or record-window JSON CLI.
- `src/contracts/mlops.py` and `src/contracts/mlops_observations.py`: validated public contracts.
- `tests/unit/test_mlops_health.py`, `tests/unit/test_mlops_window_builder.py` and
  `tests/integration/test_mlops_api.py`: behavior verification.

Next: trusted production record-source adapters and persisted baseline lookup; threshold calibration
with actual fraud data; durable evidence/results and Kafka orchestration; reproducible retraining and
champion/challenger evaluation; controlled promotion through the orchestrator and Agent 4.
