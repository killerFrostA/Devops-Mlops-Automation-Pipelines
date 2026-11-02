"""Run Agent 2 on a file of Agent 1 ANOMALY_DETECTED events.

The input file is what Agent 1 hands over: one JSON envelope, a JSON array of envelopes, or
JSONL (one envelope per line). Every incident in every message is diagnosed. Results are
appended as validated ``diagnosis.completed`` envelopes to a JSONL file, and a JSON result plus
a Markdown report are written per incident. No Kafka, Context Service or orchestrator is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.agents.diagnosis.contracts import DiagnosisAgentResult
from src.agents.diagnosis.envelope_builder import (
    DiagnosisEnvelopeError,
    append_diagnosis_jsonl,
)
from src.agents.diagnosis.file_runner import IncidentFile, render_markdown_report
from src.agents.diagnosis.llm_analyzer import build_analyzer_from_env
from src.agents.diagnosis.qdrant_retriever import QdrantKnowledgeRetriever
from src.agents.diagnosis.rules import AGENT1_MEMORY_GROWTH_MIN_SLOPE_MO_PER_MIN
from src.agents.diagnosis.service import DiagnosisAgent
from src.agents.diagnosis.tools import (
    ToolError,
    Toolbox,
    build_toolbox_from_env,
    load_static_toolbox,
)
from src.contracts.enums import AgentId, ResultStatus
from src.contracts.events import EventEnvelope
from src.contracts.tasks import AgentTask, TaskContext


ANOMALY_TOPIC = "monitoring.anomaly.detected"
ANOMALY_MESSAGE_TYPE = "ANOMALY_DETECTED"
DEFAULT_TASK_TIMEOUT = timedelta(minutes=5)
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class Agent1InputError(ValueError):
    """Raised when the Agent 1 file or one of its messages cannot be used."""


@dataclass(frozen=True)
class IncidentDiagnosis:
    received: EventEnvelope
    task: AgentTask
    result: DiagnosisAgentResult


@dataclass(frozen=True)
class WrittenDiagnosis:
    diagnosis: IncidentDiagnosis
    json_path: Path
    markdown_path: Path
    jsonl_written: bool


def load_agent1_events(path: Path) -> list[EventEnvelope]:
    """Read and validate Agent 1 anomaly envelopes from a JSON or JSONL file."""
    try:
        # Decode bytes directly: read_text() would turn stray "\r" into line breaks.
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise Agent1InputError(f"Cannot read Agent 1 file '{path}': {exc}") from exc
    if not text.strip():
        raise Agent1InputError(f"Agent 1 file '{path}' is empty.")

    raw_messages: list[tuple[str, Any]] = []
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        # Split on "\n" only: str.splitlines() would also break on stray "\r" characters.
        for line_number, line in enumerate(text.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                raw_messages.append((f"line {line_number}", json.loads(line)))
            except json.JSONDecodeError as exc:
                raise Agent1InputError(
                    f"Invalid JSON in Agent 1 file '{path}' at line {line_number}: {exc.msg}."
                ) from exc
    else:
        if isinstance(document, list):
            raw_messages = [(f"item {index}", item) for index, item in enumerate(document)]
        else:
            raw_messages = [("document", document)]

    events: list[EventEnvelope] = []
    for location, raw in raw_messages:
        try:
            event = EventEnvelope.model_validate(raw)
        except ValidationError as exc:
            raise Agent1InputError(
                f"Invalid Agent 1 envelope in '{path}' ({location}): {exc}"
            ) from exc
        if event.topic != ANOMALY_TOPIC or event.message_type != ANOMALY_MESSAGE_TYPE:
            raise Agent1InputError(
                f"Agent 1 message in '{path}' ({location}) is {event.topic}/"
                f"{event.message_type}; expected {ANOMALY_TOPIC}/{ANOMALY_MESSAGE_TYPE}."
            )
        events.append(event)
    return events


def _aware_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.utcoffset() is not None else None


def _change_context_signals(received: EventEnvelope, signals: dict[str, Any]) -> dict[str, Any]:
    """Turn ``signals.change_context`` into a dated deployment event.

    The shared AnomalyDetected payload only allows extra data inside ``signals``, so the
    architecture's change_context (section 11.1) is read from there. Either
    ``deployed_at`` or ``minutes_since_deployment`` dates the deployment.
    """
    change_context = signals.get("change_context")
    if not isinstance(change_context, dict):
        return {}
    deployment_id = change_context.get("deployment_id")
    if not isinstance(deployment_id, str) or not deployment_id.strip():
        return {}

    detected_at = _aware_timestamp(signals.get("detected_at")) or received.timestamp
    changed_at = _aware_timestamp(change_context.get("deployed_at"))
    minutes = change_context.get("minutes_since_deployment")
    if (
        changed_at is None
        and isinstance(minutes, (int, float))
        and not isinstance(minutes, bool)
        and minutes >= 0
    ):
        changed_at = detected_at - timedelta(minutes=float(minutes))
    if changed_at is None:
        return {}
    return {
        "recent_change_events": [
            {
                "change_id": deployment_id,
                "description": f"Deployment {deployment_id}",
                "changed_at": changed_at.isoformat(),
            }
        ],
        # Like Agent 1 first_seen, this is a detection time, not the exact start of the rise.
        "memory_rise_started_at": detected_at.isoformat(),
    }


def _incident_tasks(received: EventEnvelope) -> list[AgentTask]:
    raw_signals = received.payload.get("signals", {})
    if not isinstance(raw_signals, dict):
        raise Agent1InputError(
            f"Agent 1 message {received.message_id}: payload.signals must be an object."
        )

    anomaly_id = received.payload.get("anomaly_id", received.correlation_id)
    deadline = received.deadline
    if deadline is None or deadline <= datetime.now(UTC):
        deadline = datetime.now(UTC) + DEFAULT_TASK_TIMEOUT

    incidents = raw_signals.get("incidents")
    per_incident: list[tuple[str, str, str | None, dict[str, Any]]] = []
    if isinstance(incidents, list) and incidents:
        for index, incident in enumerate(incidents):
            if not isinstance(incident, dict):
                raise Agent1InputError(
                    f"Agent 1 message {received.message_id}: incidents[{index}] must be an object."
                )
            incident_id = incident.get("incident_id")
            if not isinstance(incident_id, str) or not incident_id.strip():
                raise Agent1InputError(
                    f"Agent 1 message {received.message_id}: incidents[{index}].incident_id "
                    "must be a non-empty string."
                )
            # Each diagnosis sees only its own incident and category.
            signals = deepcopy(raw_signals)
            signals["incidents"] = [incident]
            category = incident.get("category")
            if isinstance(category, str):
                signals["categories"] = [category]
            service_id = incident.get("service_id")
            per_incident.append(
                (
                    incident_id,
                    service_id if isinstance(service_id, str) and service_id.strip()
                    else received.service.service_id,
                    category if isinstance(category, str) else None,
                    signals,
                )
            )
    else:
        # Architecture format (section 11.1): flat signals keyed by the correlation ID.
        signals = deepcopy(raw_signals)
        signals.update(_change_context_signals(received, signals))
        slope = signals.get("memory_slope_mb_min")
        if (
            isinstance(slope, (int, float))
            and not isinstance(slope, bool)
            and slope >= AGENT1_MEMORY_GROWTH_MIN_SLOPE_MO_PER_MIN
        ):
            signals["memory_growth_observed"] = True
        per_incident.append(
            (received.correlation_id, received.service.service_id, None, signals)
        )

    tasks = []
    for incident_id, service_id, category, signals in per_incident:
        if category is not None:
            # Agent 1 first_seen supplies the rise time; only the change event is added.
            signals.update(
                {
                    key: value
                    for key, value in _change_context_signals(received, signals).items()
                    if key == "recent_change_events"
                }
            )
        described = f"{category} anomaly" if category else "anomaly"
        tasks.append(
            AgentTask(
                task_id=f"AGENT1-{received.message_id}-{incident_id}",
                incident_id=incident_id,
                agent_id=AgentId.DIAGNOSIS,
                objective=(
                    f"Agent 1 reported {received.severity.value} {described} {anomaly_id} "
                    f"on {service_id}; identify the root cause and recommend actions."
                ),
                task_context=TaskContext(
                    service_id=service_id,
                    namespace=received.service.namespace,
                    environment=received.environment,
                    signals=signals,
                    evidence_refs=received.evidence_refs,
                ),
                deadline=deadline,
            )
        )
    return tasks


def diagnose_agent1_event(
    received: EventEnvelope,
    agent: DiagnosisAgent | None = None,
) -> list[IncidentDiagnosis]:
    """Diagnose every incident carried by one Agent 1 anomaly envelope."""
    active_agent = agent or DiagnosisAgent()
    diagnoses = []
    for task in _incident_tasks(received):
        result = asyncio.run(active_agent.handle(task))
        if not isinstance(result, DiagnosisAgentResult) or result.status is not ResultStatus.SUCCEEDED:
            raise Agent1InputError(
                f"Diagnosis failed for incident {task.incident_id} "
                f"(message {received.message_id}): {result.rationale_summary}"
            )
        diagnoses.append(IncidentDiagnosis(received=received, task=task, result=result))
    return diagnoses


def _report_incident(diagnosis: IncidentDiagnosis) -> IncidentFile:
    received, task = diagnosis.received, diagnosis.task
    context: dict[str, Any] = {
        "agent1_message_id": received.message_id,
        "agent1_severity": received.severity.value,
        "agent1_priority": received.priority.value,
    }
    for key in ("anomaly_id", "detector", "anomaly_score"):
        if key in received.payload:
            context[key] = received.payload[key]
    return IncidentFile(
        incident_id=task.incident_id,
        service_id=task.task_context.service_id,
        namespace=task.task_context.namespace,
        environment=task.task_context.environment,
        issue_summary=task.objective,
        context=context,
    )


def write_diagnosis(
    diagnosis: IncidentDiagnosis,
    output_dir: Path,
    jsonl_path: Path,
) -> WrittenDiagnosis:
    """Append the diagnosis.completed event and write per-incident JSON/Markdown reports."""
    _, jsonl_written = append_diagnosis_jsonl(diagnosis.received, diagnosis.result, jsonl_path)
    stem = _UNSAFE_FILENAME.sub(
        "_", f"{diagnosis.task.incident_id}_{diagnosis.received.message_id}"
    )
    json_path = output_dir / f"{stem}.diagnosis.json"
    markdown_path = output_dir / f"{stem}.diagnosis.md"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(diagnosis.result.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(
            render_markdown_report(diagnosis.result, _report_incident(diagnosis)),
            encoding="utf-8",
        )
    except OSError as exc:
        raise Agent1InputError(f"Cannot write diagnosis reports to '{output_dir}': {exc}") from exc
    return WrittenDiagnosis(diagnosis, json_path, markdown_path, jsonl_written)


def run_agent1_file(
    input_path: Path,
    output_dir: Path | None = None,
    jsonl_path: Path | None = None,
    agent: DiagnosisAgent | None = None,
) -> list[WrittenDiagnosis]:
    """Diagnose every incident in an Agent 1 file and write all outputs."""
    target_dir = output_dir or input_path.parent / f"{input_path.stem}.diagnoses"
    events_path = jsonl_path or target_dir / "diagnosis.completed.jsonl"
    active_agent = agent or DiagnosisAgent()
    written = []
    for received in load_agent1_events(input_path):
        for diagnosis in diagnose_agent1_event(received, active_agent):
            written.append(write_diagnosis(diagnosis, target_dir, events_path))
    return written


def build_agent(
    *,
    tool_responses: Path | None = None,
    live_tools: bool = False,
    rag: str = "local",
    qdrant_url: str | None = None,
    qdrant_path: Path | None = None,
    llm: bool = False,
) -> DiagnosisAgent:
    """Assemble the agent from optional tools, retriever and language model."""
    toolbox: Toolbox | None = None
    if tool_responses is not None:
        toolbox = load_static_toolbox(tool_responses)
    elif live_tools:
        toolbox = build_toolbox_from_env()
        if toolbox is None:
            raise Agent1InputError(
                "--live-tools needs at least one of PIDS_PROMETHEUS_URL, PIDS_LOKI_URL, "
                "PIDS_DIAGNOSIS_KUBECTL or PIDS_DIAGNOSIS_GIT_REPO."
            )

    retriever = None
    if qdrant_url or qdrant_path:
        retriever = QdrantKnowledgeRetriever.connect(url=qdrant_url, path=qdrant_path)
    elif rag == "qdrant":
        retriever = QdrantKnowledgeRetriever.in_memory()

    analyzer = None
    if llm:
        analyzer = build_analyzer_from_env(toolbox, {**os.environ, "PIDS_DIAGNOSIS_LLM": "claude"})
    return DiagnosisAgent(retriever=retriever, analyzer=analyzer, toolbox=toolbox)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Agent 1 ANOMALY_DETECTED events read from a JSON or JSONL file."
    )
    parser.add_argument("agent1_file", type=Path, help="Agent 1 events (JSON object, array or JSONL).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for reports (defaults to <input>.diagnoses beside the input).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="diagnosis.completed JSONL path (defaults to <output-dir>/diagnosis.completed.jsonl).",
    )
    tools = parser.add_mutually_exclusive_group()
    tools.add_argument(
        "--tool-responses",
        type=Path,
        help="Recorded tool responses (JSON) replayed instead of live systems.",
    )
    tools.add_argument(
        "--live-tools",
        action="store_true",
        help="Query Prometheus/Loki/kubectl/git configured through PIDS_* variables (read-only).",
    )
    parser.add_argument(
        "--rag",
        choices=("local", "qdrant"),
        default="local",
        help="local: lexical search of the JSON corpus; qdrant: in-memory Qdrant vector search.",
    )
    parser.add_argument("--qdrant-url", help="Use an indexed Qdrant server instead.")
    parser.add_argument("--qdrant-path", type=Path, help="Use an indexed local Qdrant store instead.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Add Claude's supplemental review (needs ANTHROPIC_API_KEY; billed API calls).",
    )
    args = parser.parse_args(argv)

    try:
        agent = build_agent(
            tool_responses=args.tool_responses,
            live_tools=args.live_tools,
            rag=args.rag,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path,
            llm=args.llm,
        )
        written = run_agent1_file(args.agent1_file, args.output_dir, args.output_jsonl, agent)
    except (Agent1InputError, DiagnosisEnvelopeError, ToolError) as exc:
        print(f"Agent 1 diagnosis error: {exc}", file=sys.stderr)
        return 2

    for item in written:
        result = item.diagnosis.result
        diagnosis = result.diagnosis
        print(f"Incident {result.incident_id} (Agent 1 message {item.diagnosis.received.message_id})")
        print(
            f"  Root cause: {diagnosis.primary_root_cause} (heuristic score {result.confidence}, "
            f"evidence strength {result.evidence_strength})"
        )
        for call in result.tool_calls:
            print(f"  - tool {call.tool}.{call.operation}: {call.status}")
        for hypothesis in diagnosis.hypotheses:
            print(f"  - hypothesis {hypothesis.rank}: {hypothesis.code} ({hypothesis.probability:.2f})")
        for recommendation in result.explainability.recommendation_explanations:
            print(f"  - recommendation: {recommendation.action}")
        for document in result.knowledge.similar_incidents:
            print(
                f"  - similar incident: {document.reference_id} "
                f"({document.root_cause}, score {document.score:.2f})"
            )
        print(f"  Report: {item.markdown_path}")
        print(f"  JSONL: {'appended' if item.jsonl_written else 'duplicate skipped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
