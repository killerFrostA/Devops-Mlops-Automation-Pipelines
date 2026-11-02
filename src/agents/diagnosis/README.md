# Agent 2 · Diagnosis and root cause analysis

`DiagnosisAgent.handle()` diagnoses one bounded `AgentTask`, following section 12 of the
architecture:

1. **Targeted evidence** (optional `Toolbox`): read-only Prometheus, Loki, `kubectl get` and
   `git log` calls chosen from the incident category and any replan request (`investigation.py`).
2. **Deterministic signatures first** (`rules.py`, `signatures.py`): ranked hypotheses with the
   evidence IDs that support them.
3. **RAG** (`rag_retriever.py` lexical, or `qdrant_retriever.py` vector search): similar past
   incidents and runbooks, ordered by agreement with the leading cause.
4. **LLM tool-calling** (optional `ClaudeSupplementalAnalyzer`): Claude reviews the candidates,
   may call the same read-only tools, and may reorder close candidates. Evidence it collects is
   re-evaluated by the rules; a cause outside the rule candidates is rejected.
5. **Result**: ranked hypotheses, recommendations, limitations, `evidence_strength`, tool call
   audit, and the shared `DiagnosisCompleted` payload.

Without tools, Qdrant or an LLM, the agent runs fully offline: evidence is then derived from the
Agent 1 message only and the supplemental analyzer is a deterministic simulation. Provenance
(`provided`, `derived`, `fixture`) is kept on every evidence item; heuristic scores are not
calibrated probabilities.

| Component | Enable with | Default |
|---|---|---|
| Recorded tool responses | `--tool-responses tools.json` | off |
| Live read-only tools | `--live-tools` + `PIDS_PROMETHEUS_URL`, `PIDS_LOKI_URL`, `PIDS_DIAGNOSIS_KUBECTL` (path to kubectl, optional `PIDS_DIAGNOSIS_KUBE_CONTEXT`), `PIDS_DIAGNOSIS_GIT_REPO`, `PIDS_DIAGNOSIS_K8S_APP_LABEL` (default `app`) | off |
| Qdrant RAG | `--rag qdrant` (in-memory), `--qdrant-url` or `--qdrant-path` (indexed store) | lexical JSON search |
| Claude review | `--llm` + `ANTHROPIC_API_KEY`; `PIDS_DIAGNOSIS_LLM_MODEL` (default `claude-opus-5-5`), `PIDS_DIAGNOSIS_LLM_EFFORT` (default `high`), `PIDS_DIAGNOSIS_LLM_FALLBACKS=false` to disable server-side refusal fallbacks | deterministic simulation |

Python packages: `anthropic` (Claude) and `qdrant-client` (already in the project's `storage`
extra) are imported only when those components are used.

`TaskContext.evidence_refs` contains identifiers only. Resolved evidence content is passed to
`DiagnosisAgent.handle(task, resolved_evidence)` as `ResolvedEvidence` records defined in this
agent's own `evidence.py`, so no shared contract is changed. The records must have been requested
by the task and belong to its incident. Agent 2 has no Context Service reader. Missing IDs/content
remain unresolved and are reported as limitations. How the orchestrator resolves these IDs is
decided at integration time.

## Diagnose an Agent 1 file (main entry point)

`agent1_runner.py` reads what Agent 1 hands over — one `ANOMALY_DETECTED` envelope, a JSON array
of envelopes, or JSONL (one envelope per line) — and diagnoses **every incident of every message**:

```powershell
py -3 -m src.agents.diagnosis.agent1_runner path\to\agent1_events.jsonl --output-dir out
```

For each incident it writes `<incident>_<message>.diagnosis.json` and `.diagnosis.md`, and appends
a validated `diagnosis.completed` envelope to `out\diagnosis.completed.jsonl` (`--output-jsonl`
to change it). Re-running the same file is idempotent. Messages with another topic, invalid
envelopes and invalid JSON lines are rejected with the file location.

Two message shapes are supported, both valid against the shared `AnomalyDetected` contract:

- **Agent 1 incidents format** (current Agent 1 output): `signals.categories` and
  `signals.incidents[]` with `facts`. A message with several incidents gives one diagnosis per
  incident, each with its own `correlation_id`.
- **Architecture format** (section 11.1): flat signals such as `memory_pct`, `latency_p95_ms`,
  `error_rate`, `pod_restarts`, `memory_slope_mb_min`, keyed by the envelope `correlation_id`.

The shared payload only accepts extra data inside `signals`, so deployment context is read from
`signals.change_context = {"deployment_id": ..., "minutes_since_deployment": ...}` (or
`"deployed_at"`: ISO timestamp), dated against `signals.detected_at` or the envelope timestamp.
This is the field to agree on with Agent 1: without it no post-deployment leak, and therefore no
rollback, can be inferred.

## Causes and recommendations

| Root cause | When | Recommendations (heuristic score) |
|---|---|---|
| `MEMORY_LEAK_POST_DEPLOY` | high memory/failure + dated deployment 5 min to 6 h before the memory rise | `ROLLBACK` (0.85, with `deployment_id`), collect heap profile (0.60), `SCALE_REPLICAS` as temporary alternative (0.40) |
| `MEMORY_LIMIT_REACHED` | Agent 1 `memory_oom`, peak ratio ≥ 0.95 | `INCREASE_MEMORY_LIMIT` (0.70, local only: no shared `ActionType` yet), collect evidence (0.65) |
| `MEMORY_GROWTH` | Agent 1 `memory_pressure`, slope ≥ 5 Mo/min, no qualifying deployment | collect deployment timeline (0.65), `ROLLING_RESTART` as temporary containment (0.50) |
| `MEMORY_PRESSURE` | memory ≥ 70 % without dated deployment | collect evidence |
| `CONFIG_ERROR_CRASHLOOP` | CrashLoopBackOff + invalid-config log (0.85) or config change (0.70), not OOM | `ROLLBACK` of the latest dated change (0.80) or `FIX_CONFIGURATION` (local), collect config diff |
| `CRASH_LOOP_UNEXPLAINED` | CrashLoopBackOff without config or OOM evidence | collect previous container logs, config diff, exit code |
| `DEPENDENCY_UNAVAILABLE` | connection refused / timeouts in logs (0.78-0.83), or Agent 1 suspects another service (0.50) | `INVESTIGATE_DEPENDENCY` (local), no rollback or scaling of this service |
| `CONNECTION_POOL_EXHAUSTION` | pool-exhausted / too-many-connections logs | `TUNE_CONNECTION_POOL` (local), collect connection metrics |
| `TRAFFIC_SPIKE` | request rate ≥ 1.5 × its baseline | `SCALE_REPLICAS` (0.75) |
| `RESOURCE_PRESSURE_OR_CPU_STARVATION` | CPU ≥ 70 %, or error/latency signals | `SCALE_REPLICAS` when CPU supports it, otherwise collect evidence |

Error-rate and latency symptoms are shared by many failures. When another cause is established
and no CPU measurement reaches 50 % — or next to a dated post-deployment leak, whose GC and
contention raise CPU — CPU starvation stays a lower-ranked alternative (score halved). Agent 2
only recommends: the orchestrator's policy gate and HITL decide what is executed.
`DiagnosisAgentResult.confidence` (and the envelope `confidence`) is the leading hypothesis'
heuristic score, not a calibrated probability.

`requires_additional_evidence` is true when the leading code is symptom-level
(`INSUFFICIENT_EVIDENCE`, `MEMORY_GROWTH`, `MEMORY_PRESSURE`, `CRASH_LOOP_UNEXPLAINED`), its score
is below 0.55 or uncited, readings conflict, the language model disagrees strongly, or only
evidence collection can be recommended. Other missing information is listed in the explanation
without blocking. `evidence_strength` (0-1) combines the variety of evidence types citing the
leading hypothesis (three types = full) with their provenance (observed 1.0, derived 0.7, fixture
0.5); it is the `evidence_strength` input of the orchestrator's decision score (section 17).

Recommendations are data only; Agent 2 never executes actions. Recommendations without a
matching shared `ActionType` are omitted from the shared action tuple and explicitly recorded as
a limitation.

## Read-only evidence tools

`tools.py` implements section 18.1 (tool allowlisting). Each tool exposes a finite set of
operations; the `Toolbox` refuses anything else, bounds every call by the task deadline, caches
results per incident, caps calls per incident, and stops calling a tool after repeated failures
(circuit breaker). A failing tool never stops the diagnosis: it becomes a limitation.

| Tool | Operations | Access |
|---|---|---|
| `prometheus` | `memory_percent`, `cpu_percent` (relative to limits), `restart_count`, `error_rate`, `latency_p95_ms`, `request_rate` (+ 1-day baseline), `heap_usage`, `gc_pause` | HTTP GET `/api/v1/query` |
| `loki` | `error_logs` (samples, known signatures, dependency host) | HTTP GET `/loki/api/v1/query_range` |
| `kubernetes` | `pod_status`, `events`, `deployment_history` (dated rollout revisions) | `kubectl get … -o json` |
| `git` | `recent_changes` (commits and changed files) | `git log` |

Commands use fixed argument lists (no shell) and service/namespace names are validated before
any query is built. `investigation.py` chooses operations from the incident category and adds
the evidence named in a replan request (`signals.required_evidence`, e.g. `heap_usage_breakdown`,
`gc_metrics` after a rejected rollback, section 8.4). Dated rollouts and commits become change
events, so a deployment found by Kubernetes can establish a post-deployment leak. Recorded
responses (`--tool-responses`, format `{"tools": {"<tool>": {"<operation>": [evidence...]}}}`)
replay the same path offline with `fixture` provenance.

## Claude supplemental review

`ClaudeSupplementalAnalyzer` (in `llm_analyzer.py`) runs a manual tool-use loop on the Messages
API with `claude-opus-5-5`, `output_config.effort` and a strict JSON answer schema whose
`selected_cause` enum is the rule candidates plus `NONE`. Claude can call only the configured
read-only tools; each call is audited with `requested_by: llm`. Server-side refusal fallbacks
(`fallbacks: "default"`) are on by default. The answer is advisory: evidence it collected is
re-run through the rules, it may reorder candidates whose rule scores differ by at most 0.15,
and a stronger disagreement keeps the rule ranking and sets `requires_additional_evidence`.
A refusal, a truncated or invalid answer, a cause outside the candidates, a missing package or
missing credentials all fall back to the deterministic result with a limitation. Calls are billed.

## Run a diagnosis from a local JSON file

Run the standalone entry point from the repository root:

```powershell
py -3 -m src.agents.diagnosis.file_runner src\agents\diagnosis\examples\memory_pressure.json
```

The command validates the input and writes `<input>.diagnosis.json` and
`<input>.diagnosis.md` beside it. Use `--output-dir PATH` to choose another directory.
Invalid JSON, invalid fields, and diagnosis failures are reported to stderr with a nonzero exit
code; reports are not written for invalid input.

Required fields are `incident_id`, `service_id`, `namespace`, `environment` (`local`, `test`,
`staging`, or `production`), and `issue_summary`. Optional `signals` is a JSON object for measured
values such as `memory_percent`, `cpu_percent`, `error_rate`, `latency_ms`, and `restart_count`.
Optional `recent_changes`, `constraints`, and free-form JSON `context` provide additional context.
`evidence` is a list of existing Agent 2 `EvidenceItem` records: use `evidence_type: "log"` for
errors/log observations, `"metric"` for measured metric records, and `"k8s_event"` for alerts or
events. Each item requires `evidence_id`, `evidence_type`, `source`, and `summary`; `observed_at`,
`value`, and `confidence` are optional. Timestamps, when supplied, must be timezone-aware.
`evidence_refs` may list external IDs, but IDs without corresponding content are not resolved by
this standalone runner and are reported as limitations.

The output uses the existing `DiagnosisAgentResult` and `DiagnosisCompleted` contracts. Evidence
derived from input signals is labeled as derived, local knowledge results are reference context,
and hypothesis scores remain uncalibrated heuristics. This local workflow does not query
operational systems, Qdrant, or a language model.

## Replay Agent 1 event fixtures

Replay the two local Agent 1 `ANOMALY_DETECTED` envelopes through
`DiagnosisAgent.handle()` and append validated `DIAGNOSIS_COMPLETED` events to
a local JSONL file:

```powershell
py -3 -m src.agents.diagnosis.run_demo --output-jsonl "$env:TEMP\diagnoses.jsonl"
```

The demo extracts the incident ID from the single entry in
`payload.signals.incidents`, passes `payload.signals` as the task's signal
context, and preserves the received `message_id` as the output event's
`causation_id` and part of its idempotency key. Replaying the same message is
deduplicated; the two fixtures have distinct message IDs and each gets its own
output line. No external evidence resolver is configured, so the demo reports
the fixture's evidence references as unresolved while still using facts embedded
in the message to construct derived evidence.

## Local knowledge corpus

The retrievers read `knowledge/incidents.json` and `knowledge/runbooks.json` beside this module
(handed over with Agent 1's material), plus any `incidents_*.json` / `runbooks_*.json` files —
`incidents_agent2.json` and `runbooks_agent2.json` add Agent 2's own examples for the crash-loop,
dependency, traffic-spike and memory-limit causes. All files use UTF-8 JSON with a top-level `format_version` set to `1` and a `documents`
array. Each document has a unique `reference_id`, a non-empty `summary`, a non-empty list of unique
`tags`, a `root_cause` string for incidents (or `null` for runbooks), and an `origin` identifying
the entry as `synthetic_example` or `repository_source`. The records currently included are
synthetic examples, not real incident reports or operational runbooks.

The retriever validates both files when it is constructed. Missing files, malformed JSON,
unsupported format versions, invalid fields, and duplicate identifiers raise explicit errors;
diagnosis reports a retrieval limitation if loading or retrieval fails and continues using the
deterministic diagnosis rules. Retrieval uses case-insensitive lexical token overlap: its score is
the fraction of a document's tags matching query tokens. Results sort by descending score and then
by `reference_id`; `limit` applies separately to incidents and runbooks. Empty queries and
non-positive limits return no results.

Retrieved documents are knowledge references only. They are reported in `knowledge_refs` and are
never added to incident evidence or hypothesis `evidence_refs`; those evidence references must
come from the current incident's supplied or derived evidence.

How the diagnosis uses this retrieval (RAG):

1. The query combines the issue summary, measured signal names, evidence summaries, the
   candidate root causes (`MEMORY_LIMIT_REACHED` becomes "memory limit reached"; the leading one
   counts twice) and the Agent 1 categories. Agent 1 bookkeeping keys are left out.
2. Up to 8 candidates are retrieved, then re-ordered by root-cause agreement — same cause as the
   leading hypothesis, a related cause (`RELATED_ROOT_CAUSES` in `result_builder.py`, e.g.
   `MEMORY_LIMIT_REACHED` ↔ `INSUFFICIENT_MEMORY_LIMIT`), then agreement with an alternative —
   and finally by retrieval score; the top 3 of each kind are kept.
3. Matching incidents are named in the rationale, and the full documents (summary, past root
   cause, score) are kept in `DiagnosisAgentResult.knowledge` and rendered in the Markdown report
   with the applicable runbooks.

### Qdrant vector search

`qdrant_retriever.py` indexes the same corpus into the collections named by the architecture
(`incident_summaries`, `runbooks`) and searches them by cosine similarity. It works with the
Qdrant server of `compose.yaml` (profile `intelligence`, `PIDS_QDRANT_URL`), a local on-disk
store, or an in-memory store (`--rag qdrant`, no server needed):

```powershell
py -3 -m src.agents.diagnosis.knowledge_index --qdrant-url http://localhost:6333 --query "oom memory limit"
py -3 -m src.agents.diagnosis.agent1_runner agent1.jsonl --qdrant-url http://localhost:6333
```

Vectors come from `HashingEmbedder`, a deterministic, dependency-free embedding of words and
word pairs: it matches shared vocabulary, not paraphrases. A neural embedding model can replace
it through the `Embedder` protocol, after re-indexing. Results carry `provenance: "qdrant"`;
documents under a 0.05 similarity are dropped.

The local JSON result also contains an `explainability` section. It classifies supplied evidence,
derived evidence, and fixture context separately; lists missing information and conflicting
threshold readings; and records each hypothesis' explanation, supporting/opposing evidence
references, and heuristic ranking reason. Recommendation explanations carry their evidence
references and caveats. This section belongs to the local `DiagnosisAgentResult` envelope and
does not change the shared `DiagnosisCompleted` schema. The Markdown report renders the same
explanation data. Missing information is not treated as evidence against a hypothesis.

Rule scores and thresholds remain deterministic heuristics, not calibrated probabilities.
`MEMORY_PRESSURE` retains the existing 70% path for direct percentage inputs.
`MEMORY_GROWTH` is emitted for Agent 1 `memory_pressure` when the adapter marks growth observed
and the supplied `statistical.slope_per_minute` is at least
`AGENT1_MEMORY_GROWTH_MIN_SLOPE_MO_PER_MIN` (5 Mo/min). Its evidence references identify the
Agent 1 current value, baseline median, and slope; no memory percentage is inferred from Mo.
`MEMORY_LIMIT_REACHED` is emitted for Agent 1 `memory_oom` when
`events.memory_peak_ratio` is at least `AGENT1_MEMORY_OOM_MIN_PEAK_RATIO` (0.95); the evidence
references identify the ratio and restart count when supplied.

`MEMORY_LEAK_POST_DEPLOY` remains strict. It requires the existing high-memory/failure evidence,
a dated event identifiable as a deployment, and that event must precede the memory `first_seen`
or rise time by at least `AGENT1_DEPLOYMENT_DETECTION_MARGIN` (5 minutes). `first_seen` remains
a detection timestamp and is not treated as the exact start of memory growth. A deployment less
than 5 minutes before detection leaves the diagnosis at `MEMORY_GROWTH` and reports uncertain
correlation. Without dated deployment history, `MEMORY_GROWTH` requires additional evidence
including a deployment timeline. No rollback is recommended without dated deployment evidence.
Plain-text `recent_changes` do not qualify. The optional file fields are `memory_growth_observed`,
`memory_rise_started_at`, and `recent_change_events` (each event has `change_id`, `description`,
and timezone-aware `changed_at`). A high error rate alone does not justify scaling without CPU
or correlated latency evidence. Conflicting metric readings that cross a rule threshold are
surfaced and excluded from that metric's candidate scoring.

## Agent 1 anomaly signal adaptation

`signal_adapter.py` adapts Agent 1 anomaly `payload.signals` at the diagnosis
entry point. Original values are retained under `agent1_context`; unsupported
units, conflicting readings, and unrecognized categories or keys are reported
in diagnosis limitations. The adapter does not add deployment information or
timestamped change events.

Both `tests/fixtures/agent1_memory_oom.json` and
`tests/fixtures/agent1_memory_pressure.json` contain the exact Agent 1 event
messages supplied for local replay. Their original message IDs are retained.

### Rules that cannot fully use current Agent 1 data

- **Memory percentage:** `memory_oom` provides `events.memory_peak_ratio`, which
  the adapter converts to a percentage. `memory_pressure` provides memory in
  Mo but no memory limit, so the existing percentage threshold cannot be
  evaluated. The approved Agent 1 slope rule detects growth without inventing
  a percentage. `memory_oom` uses the separate 0.95 ratio threshold.
- **CPU percentage:** CPU usage in cores cannot be converted to a percentage
  without the applicable CPU limit; no CPU-in-cores rule is added. Limitation
  remains visible. Minimal future data change: include the CPU limit/quota and
  calculate `cores / limit`, or approve an explicit cores-based threshold.
- **Post-deployment memory leak:** Agent 1's current messages carry no
  deployment/change event. Its `first_seen` is a detection time, not an exact
  memory-rise start. A deployment event must precede `first_seen` by at least five
  minutes; shorter intervals are reported as uncertain. Minimal future data change:
  Agent 1 adds `signals.change_context` (see "Diagnose an Agent 1 file"), which the
  runner already reads; otherwise `MEMORY_GROWTH` is retained and that history is
  requested.

Recommendations are intents only and have no execution side effects. Their
verification profile is `<service_id>-slo-default`. No shared contract, other
agent, or dependency is changed by these diagnosis rules.

## Emit diagnosis.completed locally

`envelope_builder.py` wraps the shared `DiagnosisCompleted` payload in an
`EventEnvelope`, validates the payload against
`contracts/jsonschema/v1/DiagnosisCompleted.schema.json`, then appends one JSON
object per line. It does not publish to Kafka. Existing idempotency keys are
not appended again; malformed existing JSONL records and schema failures are
reported without writing a new event.

Replay both Agent 1 fixtures with an explicit output path from the repository
root:

```powershell
py -3 -m src.agents.diagnosis.run_demo --output-jsonl "$env:TEMP\diagnosis-demo.jsonl"
```

The idempotency key contains both the incident ID and the received Agent 1
`message_id`: distinct messages for the same incident therefore produce
distinct diagnosis records, while replaying the same message is deduplicated.
The `memory_pressure` fixture has its own unique `message_id` to represent a
distinct Agent 1 event. A missing or empty received `message_id` is rejected;
the writer never silently falls back to an incident-only key.

## RCA evaluation

`evaluation/scenarios/` holds 12 labeled scenarios built from the architecture's suggested
scenarios (section 18.1) and Agent 1's real messages: post-deployment memory leak (from the
message alone and found by Kubernetes tools), memory limit reached, memory growth with an old
deployment, CPU load spike, CrashLoopBackOff from invalid configuration, dependency outage,
connection pool exhaustion, scaling versus rollback, insufficient evidence, deployment too close
to detection, and a replan requesting heap/GC evidence. Each one states the expected root cause,
acceptable alternatives, expected first action, forbidden (unsafe) actions and, when relevant,
the tool calls a replan must trigger.

```powershell
py -3 -m src.agents.diagnosis.evaluation.evaluate --output-dir out\evaluation
py -3 -m src.agents.diagnosis.evaluation.evaluate --output-dir out\evaluation --llm   # billed
```

The report (`evaluation_report.md` / `.json`) compares configurations on the section 18
metrics: RCA accuracy (top-1, acceptable, top-3), first-action accuracy, unsafe recommendation
rate, additional-evidence rate, mean evidence strength, RAG hit rate, requested-evidence
collection, decision latency, and message reliability (one Agent 1 message delivered twice must
produce one `diagnosis.completed`). Results at the time of writing:

| Metric | `rules` | `rules+tools` | `rules+tools+qdrant` |
|---|---:|---:|---:|
| RCA accuracy (top-1) | 67 % | 100 % | 100 % |
| First action correct | 67 % | 100 % | 100 % |
| Unsafe recommendations | 0 % | 0 % | 0 % |
| Additional evidence requested | 58 % | 25 % | 25 % |
| RAG hit (relevant past incident first) | 73 % | 100 % | 100 % |

These scenarios were written together with the rules, so they measure coverage and regressions,
not accuracy on unseen incidents; replace or extend them with real incidents as they occur. The
`full+llm` column appears only with `--llm` and needs Anthropic API credentials.
