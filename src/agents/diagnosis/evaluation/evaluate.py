"""Evaluate Agent 2 on labeled scenarios and compare configurations (architecture section 18).

    py -3 -m src.agents.diagnosis.evaluation.evaluate --output-dir out\\evaluation
    py -3 -m src.agents.diagnosis.evaluation.evaluate --output-dir out\\evaluation --llm

Each configuration ("mode") diagnoses every scenario. Modes add capabilities one at a time so
their contribution is measurable against the deterministic baseline:

- ``rules``: deterministic rules on the Agent 1 message only, lexical local RAG;
- ``rules+tools``: plus the scenario's recorded read-only tool responses;
- ``rules+tools+qdrant``: plus Qdrant vector retrieval (in-memory store);
- ``full+llm`` (``--llm`` only): plus Claude's supplemental review — billed API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agents.diagnosis.agent1_runner import diagnose_agent1_event
from src.agents.diagnosis.contracts import DiagnosisAgentResult
from src.agents.diagnosis.envelope_builder import append_diagnosis_jsonl
from src.agents.diagnosis.llm_analyzer import ClaudeSupplementalAnalyzer
from src.agents.diagnosis.qdrant_retriever import QdrantKnowledgeRetriever
from src.agents.diagnosis.result_builder import RELATED_ROOT_CAUSES
from src.agents.diagnosis.service import DiagnosisAgent
from src.agents.diagnosis.tools import static_toolbox
from src.contracts.events import EventEnvelope


SCENARIO_DIR = Path(__file__).with_name("scenarios")
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
COLLECT = "COLLECT_MORE_EVIDENCE"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expectation(_Model):
    root_cause: str
    acceptable_root_causes: tuple[str, ...] = ()
    first_action: str
    forbidden_actions: tuple[str, ...] = ()
    required_tool_calls: tuple[str, ...] = ()


class Scenario(_Model):
    scenario_id: str
    title: str
    architecture_ref: str
    agent1_fixture: str | None = None
    event: dict[str, Any] | None = None
    signals_patch: dict[str, Any] = Field(default_factory=dict)
    tool_responses: dict[str, dict[str, list[dict[str, Any]]]] = Field(default_factory=dict)
    expected: Expectation


@dataclass(frozen=True)
class Mode:
    name: str
    tools: bool
    qdrant: bool
    llm: bool


MODES = (
    Mode("rules", tools=False, qdrant=False, llm=False),
    Mode("rules+tools", tools=True, qdrant=False, llm=False),
    Mode("rules+tools+qdrant", tools=True, qdrant=True, llm=False),
)
LLM_MODE = Mode("full+llm", tools=True, qdrant=True, llm=True)


@dataclass
class Outcome:
    scenario_id: str
    mode: str
    predicted: str
    expected: str
    top1_correct: bool
    acceptable_correct: bool
    top3_correct: bool
    first_action: str
    action_correct: bool
    unsafe_actions: list[str]
    needs_more_evidence: bool
    evidence_strength: float
    rag_hit: bool | None
    required_tools_ok: bool | None
    latency_ms: float
    error: str | None = None


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    return [
        Scenario.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def _shorthand_envelope(event: dict[str, Any]) -> dict[str, Any]:
    incident_id = event["incident_id"]
    return {
        "message_id": f"MSG-{incident_id}",
        "correlation_id": incident_id,
        "trace_id": f"TRACE-{incident_id}",
        "source": "monitoring-agent",
        "target": "orchestrator",
        "topic": "monitoring.anomaly.detected",
        "message_type": "ANOMALY_DETECTED",
        "priority": event.get("priority", "P1"),
        "severity": event.get("severity", "HIGH"),
        "timestamp": event.get("detected_at", "2026-10-20T10:00:00Z"),
        "environment": event.get("environment", "production"),
        "service": {
            "service_id": event["service_id"],
            "namespace": event.get("namespace", "prod"),
            "version": event.get("version", "unknown"),
        },
        "idempotency_key": f"monitoring-agent:anomaly:{incident_id}",
        "payload": {
            "anomaly_id": f"ANM-{incident_id}",
            "detector": "evaluation-scenario",
            "anomaly_score": 0.9,
            "signals": event["signals"],
        },
    }


def scenario_event(scenario: Scenario) -> EventEnvelope:
    if scenario.agent1_fixture is not None:
        raw = json.loads((FIXTURE_DIR / scenario.agent1_fixture).read_text(encoding="utf-8"))
    elif scenario.event is not None:
        raw = _shorthand_envelope(scenario.event)
    else:
        raise ValueError(f"Scenario {scenario.scenario_id} has neither agent1_fixture nor event")
    raw = deepcopy(raw)
    raw["payload"]["signals"].update(deepcopy(scenario.signals_patch))
    # Scenarios replay past events; the task deadline must still be in the future.
    raw["deadline"] = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    return EventEnvelope.model_validate(raw)


def _actions(result: DiagnosisAgentResult) -> list[str]:
    return [item.action for item in result.explainability.recommendation_explanations]


def _first_action(actions: list[str]) -> str:
    return next((action for action in actions if action != COLLECT), COLLECT if actions else "NONE")


def _rag_hit(result: DiagnosisAgentResult, expected: str) -> bool | None:
    incidents = result.knowledge.similar_incidents
    if not incidents or expected == "INSUFFICIENT_EVIDENCE":
        return None
    top = incidents[0].root_cause
    return top == expected or top in RELATED_ROOT_CAUSES.get(expected, frozenset())


def evaluate_scenario(
    scenario: Scenario,
    mode: Mode,
    qdrant: QdrantKnowledgeRetriever | None,
) -> Outcome:
    toolbox = static_toolbox(scenario.tool_responses) if mode.tools and scenario.tool_responses else None
    agent = DiagnosisAgent(
        retriever=qdrant if mode.qdrant else None,
        analyzer=ClaudeSupplementalAnalyzer(toolbox=toolbox) if mode.llm else None,
        toolbox=toolbox,
    )
    expected = scenario.expected
    started = perf_counter()
    try:
        diagnosis = diagnose_agent1_event(scenario_event(scenario), agent)[0].result
    except Exception as exc:  # A failing scenario is reported, not fatal for the run.
        return Outcome(
            scenario.scenario_id, mode.name, "ERROR", expected.root_cause, False, False, False,
            "NONE", False, [], True, 0.0, None, None, (perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = (perf_counter() - started) * 1000
    codes = [hypothesis.code for hypothesis in diagnosis.diagnosis.hypotheses]
    accepted = {expected.root_cause, *expected.acceptable_root_causes}
    actions = _actions(diagnosis)
    first_action = _first_action(actions)
    succeeded_calls = {
        f"{call.tool}.{call.operation}" for call in diagnosis.tool_calls if call.status == "succeeded"
    }
    return Outcome(
        scenario_id=scenario.scenario_id,
        mode=mode.name,
        predicted=codes[0],
        expected=expected.root_cause,
        top1_correct=codes[0] == expected.root_cause,
        acceptable_correct=codes[0] in accepted,
        top3_correct=bool(accepted & set(codes[:3])),
        first_action=first_action,
        action_correct=first_action == expected.first_action,
        unsafe_actions=sorted(set(actions) & set(expected.forbidden_actions)),
        needs_more_evidence=diagnosis.diagnosis.requires_additional_evidence,
        evidence_strength=diagnosis.evidence_strength,
        rag_hit=_rag_hit(diagnosis, expected.root_cause),
        required_tools_ok=(
            set(expected.required_tool_calls) <= succeeded_calls
            if expected.required_tool_calls and mode.tools
            else None
        ),
        latency_ms=latency_ms,
    )


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    rag = [item.rag_hit for item in outcomes if item.rag_hit is not None]
    required = [item.required_tools_ok for item in outcomes if item.required_tools_ok is not None]
    return {
        "scenarios": len(outcomes),
        "errors": sum(item.error is not None for item in outcomes),
        "rca_accuracy_top1": _rate([item.top1_correct for item in outcomes]),
        "rca_accuracy_acceptable": _rate([item.acceptable_correct for item in outcomes]),
        "rca_accuracy_top3": _rate([item.top3_correct for item in outcomes]),
        "first_action_accuracy": _rate([item.action_correct for item in outcomes]),
        "unsafe_recommendation_rate": _rate([bool(item.unsafe_actions) for item in outcomes]),
        "needs_more_evidence_rate": _rate([item.needs_more_evidence for item in outcomes]),
        "mean_evidence_strength": round(mean(item.evidence_strength for item in outcomes), 3),
        "rag_hit_rate": _rate(rag),
        "requested_evidence_collected_rate": _rate(required),
        "mean_decision_latency_ms": round(mean(item.latency_ms for item in outcomes), 1),
    }


def duplicate_delivery_check(scenario: Scenario) -> dict[str, Any]:
    """Message reliability: the same Agent 1 message delivered twice yields one event."""
    received = scenario_event(scenario)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "diagnosis.completed.jsonl"
        written = []
        for _ in range(2):
            diagnosis = diagnose_agent1_event(received)[0].result
            written.append(append_diagnosis_jsonl(received, diagnosis, output)[1])
        lines = len(output.read_text(encoding="utf-8").splitlines())
    return {
        "scenario_id": scenario.scenario_id,
        "deliveries": 2,
        "events_written": lines,
        "idempotent": written == [True, False] and lines == 1,
    }


def run_evaluation(scenarios: list[Scenario], with_llm: bool = False) -> dict[str, Any]:
    modes = (*MODES, LLM_MODE) if with_llm else MODES
    qdrant = QdrantKnowledgeRetriever.in_memory()
    outcomes = {mode.name: [evaluate_scenario(item, mode, qdrant) for item in scenarios] for mode in modes}
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "modes": [mode.name for mode in modes],
        "metrics": {name: summarize(items) for name, items in outcomes.items()},
        "outcomes": {name: [asdict(item) for item in items] for name, items in outcomes.items()},
        "reliability": duplicate_delivery_check(scenarios[0]),
        "scenarios": {item.scenario_id: item.title for item in scenarios},
    }


_METRIC_LABELS = {
    "rca_accuracy_top1": "Précision RCA (top-1)",
    "rca_accuracy_acceptable": "Précision RCA (causes acceptables)",
    "rca_accuracy_top3": "Bonne cause dans le top-3",
    "first_action_accuracy": "Première action correcte",
    "unsafe_recommendation_rate": "Recommandations dangereuses (↓)",
    "needs_more_evidence_rate": "Demande de preuves supplémentaires",
    "mean_evidence_strength": "Force moyenne des preuves",
    "rag_hit_rate": "RAG : incident passé pertinent en tête",
    "requested_evidence_collected_rate": "Preuves demandées collectées (replan)",
    "mean_decision_latency_ms": "Latence moyenne de décision (ms)",
    "errors": "Erreurs",
}


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value <= 1 and not isinstance(value, bool):
        return f"{value:.0%}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    modes = report["modes"]
    lines = [
        "# Évaluation de l’agent 2 (diagnostic RCA)",
        "",
        f"Générée le {report['generated_at']} sur {len(report['scenarios'])} scénarios étiquetés.",
        "",
        "## Métriques par configuration",
        "",
        "| Métrique | " + " | ".join(f"`{mode}`" for mode in modes) + " |",
        "|---|" + "---:|" * len(modes),
    ]
    for key, label in _METRIC_LABELS.items():
        values = []
        for mode in modes:
            value = report["metrics"][mode][key]
            values.append(f"{value}" if key in ("mean_decision_latency_ms", "errors") else _cell(value))
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(["", "## Résultat par scénario", ""])
    lines.append("| Scénario | Attendu | " + " | ".join(f"`{mode}`" for mode in modes) + " |")
    lines.append("|---|---|" + "---|" * len(modes))
    by_mode = {mode: {item["scenario_id"]: item for item in report["outcomes"][mode]} for mode in modes}
    for scenario_id in report["scenarios"]:
        expected = by_mode[modes[0]][scenario_id]["expected"]
        cells = []
        for mode in modes:
            item = by_mode[mode][scenario_id]
            mark = "✅" if item["acceptable_correct"] else "❌"
            unsafe = " ⚠️ " + ", ".join(item["unsafe_actions"]) if item["unsafe_actions"] else ""
            cells.append(f"{mark} {item['predicted']} → {item['first_action']}{unsafe}")
        lines.append(f"| {scenario_id} | {expected} | " + " | ".join(cells) + " |")
    reliability = report["reliability"]
    lines.extend(
        [
            "",
            "## Fiabilité des messages",
            "",
            f"Le même message de l’agent 1 livré {reliability['deliveries']} fois a produit "
            f"{reliability['events_written']} événement(s) `diagnosis.completed` "
            f"({'idempotent' if reliability['idempotent'] else 'NON idempotent'}).",
            "",
            "Les scores sont heuristiques; les scénarios sont synthétiques et les réponses "
            "d’outils sont enregistrées, pas issues de systèmes réels.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Agent 2 RCA on labeled scenarios.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also evaluate with Claude (needs ANTHROPIC_API_KEY; billed API calls).",
    )
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenarios_dir)
    if not scenarios:
        print(f"No scenarios found in {args.scenarios_dir}", file=sys.stderr)
        return 2
    report = run_evaluation(scenarios, with_llm=args.llm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "evaluation_report.json"
    markdown_path = args.output_dir / "evaluation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    for mode in report["modes"]:
        metrics = report["metrics"][mode]
        print(
            f"{mode:20s} RCA top-1 {_cell(metrics['rca_accuracy_top1']):>5}  "
            f"acceptable {_cell(metrics['rca_accuracy_acceptable']):>5}  "
            f"action {_cell(metrics['first_action_accuracy']):>5}  "
            f"unsafe {_cell(metrics['unsafe_recommendation_rate']):>5}"
        )
    print(f"Report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
