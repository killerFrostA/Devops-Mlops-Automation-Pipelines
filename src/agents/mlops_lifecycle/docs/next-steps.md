# Agent 6 next steps and definition of done

Agent 6 currently proves its local assessment and evaluation paths. It is
not a production-wide autonomous MLOps service. The work below is ordered
so each milestone can be reviewed and tested before the next depends on it.
No delivery dates or budget allocations are implied.

## Current baseline

| Capability | Current state |
| --- | --- |
| Binary drift and labeled performance | Working locally with fixed baseline bins, PSI, F1, delayed-label matching and configurable policy |
| Generic project evaluation | Working for binary F1, regression MAE/RMSE, optional text rubric, and trusted custom plugins |
| Project policy storage | In memory for one process; CLI registers a manifest for each run |
| Real project connection | Offline script exercised one external project; no reusable live connector |
| Candidate training | Synthetic machine-failure sklearn/MLflow demonstration only |
| OpenAI/Groq | Structured-output adapter and mocked tests; no live acceptance run |
| Orchestration and promotion | Bounded `AgentTask` handler; no durable worker, approval or deployment |

## 1. Trusted project and evidence source

**Next concrete engineering milestone:** make the orchestrator send a
project/model/version and an immutable evidence reference, then have Agent 6
fetch the matching registered manifest and observation batch from trusted
stores. Do this generically, without embedding a specific project's model
loader in Agent 6.

Deliverables:

- Define `ProjectManifestRepository` and `EvaluationBatchRepository`
  ports with tenant/project/model/version and time-window scoping.
- Implement a local durable adapter for integration testing, then a
  production-backed adapter selected by deployment configuration.
- Store immutable batch bytes and their digest; verify identity, digest,
  schema version, ownership and requested evidence scope on every read.
- Change the orchestrator path to pass references instead of raw prediction
  or text samples in `task_context.signals`.
- Add failure tests for missing versions, unauthorized access, altered
  evidence, duplicate IDs, stale records and unavailable storage.

**Done when:** restarting the process does not lose a project registration;
two different projects can be assessed from stored evidence; the same task
always resolves to the same policy and batch; wrong or tampered references
fail without evaluating or invoking an LLM.

## 2. Production observation and outcome adapters

Create project adapters that emit versioned prediction records when the
serving model runs and later join independently confirmed outcomes. Start
with two unlike task types, such as binary classification and regression,
to prove that the integration contract is genuinely reusable. The model
application owns model loading, inference, preprocessing and its data
access; Agent 6 owns validation and evaluation after ingestion.

Require stable observation IDs, timezone-aware event and outcome times,
deployed model version, cohort/window identity, feature/baseline version,
label confirmation cutoff and reproducible evidence references. Reject
mixed model versions, duplicate IDs, missing provenance and comparisons
that reuse training data as a purported held-out test.

**Done when:** a real, read-only model project can produce a durable
evaluation batch from its serving logs and later outcomes, without manual
copying of rows into the Agent 6 CLI. Integration tests cover delayed
labels and backfills.

## 3. Health coverage and calibrated policy per task

Extend the monitoring contract beyond today's binary F1 and fixed PSI
windows. Define task-specific quality and drift measures, including
regression error changes and text quality assessed with human-reviewed
references. Keep an explicit `INSUFFICIENT_EVIDENCE` state when outcomes
are sparse, biased, late or missing. Version baselines, feature
transformations, rubric sets and policy thresholds.

Calibrate thresholds on representative historical windows and measure
alert precision, delay and label coverage. Establish who owns each
project's thresholds and their change review. Do not infer degradation
from distribution drift alone.

**Done when:** the same ingestion path yields bounded, versioned health
reports for at least two task families; a replayed historical dataset
produces deterministic decisions; false-alarm and missed-degradation
tests are documented.

## 4. Candidate evaluation against the actual champion

Replace the synthetic machine-only training path with task-specific
candidate pipeline plugins. Fetch the deployed champion artifact and its
serving metadata from an authenticated registry. Use reproducible,
leakage-aware train/validation/test splits and compare champion and
candidate on the same held-out cohort, including latency and domain
constraints. Persist dataset digests, code and dependency versions,
policy, run ID and candidate artifact.

**Done when:** a candidate for a real project can be reproduced from
stored evidence and compared with the actual deployed champion. A
failed gate keeps the champion; a passed gate creates a review request,
not an automatic deployment.

## 5. Orchestrator, human review and deployment handoff

Connect the Agent 6 result to durable workflow state and event transport.
The orchestrator must distinguish task execution status from model-policy
decision, request human approval for the exact candidate/action, and
delegate deployment to Agent 4. Add canary criteria, independent
post-deployment verification and rollback triggers. Handle retries,
duplicate delivery, deadlines and expired approval without duplicate
effects.

**Done when:** sandbox exercises pass, fail, insufficient evidence,
approval, rejection, expiry, canary failure and rollback paths with
auditable actor, model, dataset, policy and action identifiers.

## 6. LLM and operations hardening

For text evaluation, approve the data classification and provider before
sending prompts, outputs, references or rubrics externally. Validate
provider/model compatibility, budget, timeout, retries, rate limits and
retention. Compare LLM judgments with a human-labeled reference set and
track disagreement; an LLM grade remains evidence for review, never
promotion authority.

Add structured logs and traces without raw secrets or sensitive text,
metrics for coverage/latency/error rate, alerting, runbooks and
least-privilege secret delivery. Keep production credentials out of the
repository and images. A local Agent 6 `.env` is for development;
production should inject managed secrets.

**Done when:** provider failures and malformed output fail closed;
a reviewer can inspect cost, drift and disagreement over time; secret,
privacy and incident-response checks pass.

## Acceptance gate for Agent 6

The agent is ready for production only after a representative project
runs through trusted observation collection, delayed outcomes, a
calibrated evaluation, actual-champion comparison, approved deployment
handoff and independent post-deployment verification. The complete
workflow needs real integration tests, operational ownership and
repeatable audit evidence.

The present local demos and mocked provider tests do not satisfy that
gate. See [Architecture](architecture.md) for the implemented paths and
[Project integration](integration.md) for current onboarding.
