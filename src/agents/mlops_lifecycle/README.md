# Agent 6: MLOps lifecycle

Owner: Member 6. Agent 6 assesses model health locally, trains and compares a candidate on a
synthetic machine-failure dataset, stores the run and candidate in local MLflow, and requests
promotion review when a configured gate passes. Production data connectors and deployment are
separate work.

Detailed Agent 6 documentation:

- [Current architecture and file ownership](docs/architecture.md)
- [Integrate another project, configure keys, and run examples](docs/integration.md)
- [Prioritized next steps and acceptance criteria](docs/next-steps.md)

## Project-neutral evaluation

Agent 6 can also assess **registered projects** without assuming they are fraud or machine
models. A trusted owner registers a manifest for one project, model version, task type,
evaluator and policy. A project adapter supplies a batch of deployed predictions paired with
reference outcomes and a dataset/evidence reference. The evaluator computes metrics and a
decision, then Agent 6 returns that report to the orchestrator. The orchestrator controls
durable evidence, approvals and deployment. The generic path does not load arbitrary models,
read arbitrary databases or deploy candidates by itself.

Built-in evaluator IDs are `binary-f1-v1` (binary classification), `regression-mae-v1`
(numeric regression) and `text-rubric-llm-v1` (text generation). Additional task types
use task type `custom`, a versioned `evaluator_config`, and a trusted
`ProjectEvaluator` plugin that validates its own sample payload and policy.
A manifest's model version and policy are immutable within a service instance. A batch
must match its registered model and task, contain unique observation IDs and identify
its evidence. Insufficient samples, too few positive labels or undefined binary F1 produce
`INSUFFICIENT_EVIDENCE`. `PASS` is a metric gate only: every report still requests
human review before promotion.

Run two unrelated local examples from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle.project_demo --manifest contracts/examples/agent6/projects/support-routing-manifest.json --batch contracts/examples/agent6/projects/support-routing-batch.json
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle.project_demo --manifest contracts/examples/agent6/projects/energy-demand-manifest.json --batch contracts/examples/agent6/projects/energy-demand-batch.json
```

The first reports classification F1; the second reports regression MAE and RMSE.
These four-row fixtures verify plumbing only. Their `local:` evidence references are
illustrative identifiers, not cryptographic proof or statistically representative data.
For a real project, replace the batch with records from its serving logs and confirmed
outcomes on a held-out, timestamped cohort. Store and verify that batch in the project's
evidence system. The project adapter owns model loading, inference and any CSV/database
access; Agent 6 evaluates its standardized records.

A text-generation project can opt in to an external rubric judge. Install the optional
.[llm] dependencies, then edit this Agent 6 package's own src/agents/mlops_lifecycle/.env file
(already ignored by Git). Its template is .env.example in the same folder.
The project-root .env remains for platform settings; Agent 6 reads its own file:

```dotenv
AGENT6_LLM_PROVIDER=openai
AGENT6_LLM_MODEL=<a-structured-output-model-you-can-access>
OPENAI_API_KEY=<paste-your-key-here>
```

For Groq, set AGENT6_LLM_PROVIDER=groq, select a Groq model that supports strict
structured outputs, and fill GROQ_API_KEY instead. Process environment variables
override .env; --provider and --model can override the selected provider/model.
The manifest must set allow_external_llm to true. After filling the settings, run:

```powershell
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle.project_demo --manifest contracts/examples/agent6/projects/knowledge-assistant-manifest.json --batch contracts/examples/agent6/projects/knowledge-assistant-batch.json
```

This sends each prompt, generated output, reference and rubric to the selected
provider, so use only data approved for that service. The adapter validates the
structured score. Refusals, incomplete responses and invalid scores fail the task;
LLM scores always yield REVIEW_REQUIRED and never authorize promotion. No provider
call occurs for binary, regression or local health examples. The fixture has only
two simple questions and verifies integration, not production model quality.

For orchestrator tasks, register the manifest in the trusted Agent 6 service first.
Then put an `EvaluationBatch` under `task_context.signals.project_evaluation` and
include its `evidence_ref` in `task_context.evidence_refs`. The agent rejects an
unregistered model version or out-of-scope evidence. The current registration is in
memory; production needs a durable, authenticated registry and adapters that fetch
bounded evidence by reference instead of embedding sensitive text in task signals.

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

## Test against the external fraud application

This repository's Agent 6 can run an **offline health test** against the separate
`../Fraud_Detection_MLOps` application. The command reads its actual
`Data/payment_fraud.csv` and scores two disjoint 2,000-row cohorts with the model loaded by
`API/services.py`. The scorer runs in a separate Python 3.12 environment installed from the
serving artifact's `requirements.txt`, so the application and Agent 6 keep their own dependencies.

From the Agent 6 repository root, with that model environment prepared:

```powershell
.\.venv\Scripts\python.exe scripts/test_agent6_fraud_project.py --target-root ../Fraud_Detection_MLOps --model-python artifacts/agent6-fraud-project/model-env/Scripts/python.exe
```

The tested result is in `artifacts/agent6-fraud-project/assessment/fraud-project-health-report.json`.
It reported `HEALTHY` with `Category` distribution drift: reference F1 0.408, current F1
0.389, and an absolute F1 drop of about 0.020 below the demo policy's 0.05 threshold. These
values verify that the external model, CSV and Agent 6 assessment connect. The result stores
dataset/model hashes, model runtime versions and numeric histogram boundaries. The fraud
repository is not edited by the command.

This CSV has no event IDs, event times or label-confirmation times. Its row order is not a
production monitoring window, and the saved model may have trained on these same rows. The
API returns a class but no calibrated score. Consequently this test does **not** establish
production health, unbiased model accuracy or candidate-promotion eligibility. It exercises
the health-assessment path; the existing machine demo separately exercises local candidate
training and registration. Live fraud monitoring needs recorded predictions and later outcomes
with stable IDs and real timestamps.

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

## Package layout and next work

The Agent 6 package is organized by responsibility:

- `service.py`: the Agent 6 coordinator and task boundary.
- `health/`: `assessor.py` decides model health; `metrics.py` calculates PSI and
  binary counts; `window_builder.py` aligns records and builds fixed-bin windows.
- `evaluation/`: `engine.py` registers projects and runs task-specific evaluators;
  `llm_judge.py` connects an opted-in text evaluator to OpenAI or Groq.
- `training/`: `pipeline.py` trains and compares the local candidate;
  `demo_data.py` creates synthetic history for the example.
- `adapters/`: `local_files.py` reads local JSON with content digests;
  `mlflow_registry.py` registers a local candidate in MLflow.
- `interfaces/`: `ports.py` defines the data-source, training and registry
  interfaces that production adapters must implement.
- `cli/`: `health.py`, `lifecycle_demo.py` and `project_demo.py` implement
  the three local commands. Top-level `__main__.py`, `demo.py` and
  `project_demo.py` are small wrappers that preserve the existing commands.

Shared Pydantic contracts remain in `src/contracts/` because orchestrator tasks and
other agents use the same schemas. The package tests are in `tests/unit/` and
`tests/integration/`.

The immediate next milestone is durable, authenticated project registration and evidence
lookup by reference. The full sequence and completion criteria are in
[Next steps](docs/next-steps.md).
