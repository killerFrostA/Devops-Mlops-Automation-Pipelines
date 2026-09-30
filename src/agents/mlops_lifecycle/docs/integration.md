# Integrate a project with Agent 6

Agent 6 is project-neutral at the **evaluation boundary**. A project supplies
the model identity, policy, predictions and reference outcomes in the validated
contracts. The project remains responsible for its own model loader, serving
API, CSV/database reader and feature preprocessing. Agent 6 evaluates the
prepared evidence. There is no automatic discovery of arbitrary models.

## Choose an evaluator and register a policy

Create a `ProjectManifest` for each immutable project/model version:

| Task type | Built-in evaluator | Required policy field | Input pair |
| --- | --- | --- | --- |
| `binary_classification` | `binary-f1-v1` | `min_f1` | 0/1 prediction and 0/1 reference label |
| `regression` | `regression-mae-v1` | `max_mae` | Finite numeric prediction and reference |
| `text_generation` | `text-rubric-llm-v1` | `min_text_score` | Prompt, generated output, reference, rubric |
| `custom` | Project-supplied plugin | Validated `evaluator_config` | JSON payload validated by that plugin |

All manifests name `project_id`, `model_id`, `model_version`,
`evaluator_id`, `policy_version` and `min_samples`. Binary evaluation
also has `min_positive_samples`. The text evaluator has a 100-sample batch
limit. A text manifest must opt in with `allow_external_llm: true` before
the external judge can run. Do not treat the example thresholds as calibrated
acceptance criteria.

See the example manifest/batch pairs in
[contracts/examples/agent6/projects](../../../../contracts/examples/agent6/projects).
The corresponding contract code is
[mlops_projects.py](../../../contracts/mlops_projects.py). The committed
JSON Schemas are in
[contracts/jsonschema/v1](../../../../contracts/jsonschema/v1).

## Prepare an evaluation batch

For each observation, record a stable unique ID, timezone-aware observation
time, the exact deployed model version, a prediction and an independently
obtained reference outcome. Include `dataset_ref` and `evidence_ref`
for the cohort. A real adapter should persist the input dataset and prove
that its reference points to those exact bytes. The current generic CLI
accepts a self-declared reference but does not verify it against a store.

For online models, collect prediction events when inference happens and
join confirmed outcomes later by observation ID. For an offline model, a
project-owned adapter may load its saved model and dataset, run inference
and emit the same contract. Split and cohort provenance must be checked
before using those scores as evidence: re-scoring training rows cannot
establish generalization. When labels arrive late, record their confirmation
time and assess only outcomes available at the chosen cutoff.

The binary health path uses additional contracts in
[mlops_observations.py](../../../contracts/mlops_observations.py): a saved
baseline, versioned predictions, confirmed labels and fixed histogram bins.
Generic regression and text evaluation currently do **not** have comparable
production drift-window builders. See [Next steps](next-steps.md).

## Run local, read-only examples

From the repository root, install the project and development dependencies,
then run the two project-neutral examples:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle.project_demo --manifest contracts/examples/agent6/projects/support-routing-manifest.json --batch contracts/examples/agent6/projects/support-routing-batch.json
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle.project_demo --manifest contracts/examples/agent6/projects/energy-demand-manifest.json --batch contracts/examples/agent6/projects/energy-demand-batch.json
```

The support-routing fixture reports F1; the energy-demand fixture reports
MAE and RMSE. Each has only four rows and a `local:` evidence reference.
They test the integration shape, not production quality.

For the binary health-only local source example, run:

```powershell
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle --sources contracts/examples/agent6/local-sources/machine-sources.json
```

For synthetic candidate training and MLflow registration, first install
`.[ml]` and run `python -m src.agents.mlops_lifecycle.demo`. This writes
to `artifacts/agent6-demo/`. The local HTTP API exposes only prepared
binary health input at `POST /api/v1/mlops/health/assess`; there is no
generic project-evaluation HTTP endpoint yet.

## Optional OpenAI or Groq text judge

Agent 6 reads provider settings from its own ignored
`src/agents/mlops_lifecycle/.env`. The root `.env` is for platform
settings. Copy the package-local [.env.example](../.env.example) if needed,
then fill the model and the chosen provider's key:

```dotenv
AGENT6_LLM_PROVIDER=openai
AGENT6_LLM_MODEL=<model-supporting-strict-structured-outputs>
OPENAI_API_KEY=<your-key>
GROQ_API_KEY=
```

For Groq, choose `groq`, a compatible model and `GROQ_API_KEY`.
Process environment variables override the file. CLI `--provider` and
`--model` can override those two settings; never pass an API key as a
CLI argument. Only an explicitly opted-in text manifest invokes the
external service. The request sends the prompt, generated answer, reference
and rubric. Do not send data that has not been approved for that provider.

After setup, run the small text fixture:

```powershell
.\.venv\Scripts\python.exe -m src.agents.mlops_lifecycle.project_demo --manifest contracts/examples/agent6/projects/knowledge-assistant-manifest.json --batch contracts/examples/agent6/projects/knowledge-assistant-batch.json
```

The provider returns a validated score and rationale, and Agent 6 reports
`REVIEW_REQUIRED`. Refusals, incomplete responses and invalid scores
fail the evaluation. These provider calls were **mocked** in repository
tests; a live-provider acceptance run remains to be done.

## Orchestrator and plugin integration

A trusted service first calls `MLOpsLifecycleAgent.register_project` with
a reviewed manifest. An orchestrator task then places an
`EvaluationBatch` under `task_context.signals.project_evaluation` and
includes the same `evidence_ref` in `task_context.evidence_refs`.
`handle` verifies the agent ID, deadline, evidence scope and project
registration. Its successful `AgentResult.payload` contains the
`ProjectEvaluationReport`. A policy `FAIL` or
`INSUFFICIENT_EVIDENCE` still has task status `SUCCEEDED` because the
evaluation ran; consumers must inspect the report decision.

To add a new task family, implement the `ProjectEvaluator` protocol,
validate the custom sample payload and `evaluator_config`, and register
the plugin in `ProjectEvaluationService` before registering the project.
The service checks that the plugin's report preserves the registered model,
policy, dataset, evidence reference and sample count. The stock JSON CLI
does not dynamically load third-party plugins.

For production, replace embedded raw task batches with authenticated
evidence references and a trusted fetcher. The orchestrator and Agent 4
must own approval, deployment, verification and rollback. Agent 6's
`PASS` and `REQUEST_PROMOTION_REVIEW` outputs do not authorize those
actions.

## Troubleshooting and verification

- `Project and exact model version must be registered`: register the
  manifest in the same service instance before evaluating that batch.
- `Evaluation samples do not match`: the manifest's task type and every
  sample's discriminator must agree.
- `INSUFFICIENT_EVIDENCE`: inspect minimum sample count, positive labels
  or text batch limit before changing a threshold.
- `External LLM evaluation needs explicit project opt-in`: review the
  data-sharing decision and set `allow_external_llm` only if approved.
- `AGENT6_LLM_MODEL` or API key missing: edit Agent 6's package-local
  `.env`, not the platform root file.
- Run `.\.venv\Scripts\python.exe scripts/check.py` for formatting,
  lint, strict typing, schema drift, tests and package build.

For the current internals, see [Architecture](architecture.md). For
implementation priorities and acceptance criteria, see
[Next steps](next-steps.md).
