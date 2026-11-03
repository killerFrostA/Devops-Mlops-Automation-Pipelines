"""Evaluate Agent 2 on labeled scenarios and compare configurations (architecture section 18).

    py -3 -m src.agents.diagnosis.evaluation.evaluate --output-dir out\\evaluation
    py -3 -m src.agents.diagnosis.evaluation.evaluate --output-dir out\\evaluation --llm
        --scenario-delay 3

Each configuration ("mode") diagnoses every scenario. Modes add capabilities one at a time so
their contribution is measurable against the deterministic baseline:

- ``rules``: deterministic rules on the Agent 1 message only, lexical local RAG;
- ``rules+tools``: plus the scenario's recorded read-only tool responses;
- ``rules+tools+qdrant``: plus Qdrant vector retrieval (in-memory store);
- ``full+llm`` (``--llm`` only): plus the LLM supplemental review — an OpenAI-compatible
  endpoint by default (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY_ENV), or ``--llm anthropic``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
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
from src.agents.diagnosis.llm_analyzer import LLMConfigurationError, build_analyzer_from_env
from src.agents.diagnosis.ports import SupplementalAnalyzer
from src.agents.diagnosis.qdrant_retriever import QdrantKnowledgeRetriever
from src.agents.diagnosis.result_builder import RELATED_ROOT_CAUSES
from src.agents.diagnosis.service import DiagnosisAgent
from src.agents.diagnosis.tools import Toolbox, static_toolbox
from src.contracts.events import EventEnvelope

SCENARIO_DIR = Path(__file__).with_name("scenarios")
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
COLLECT = "COLLECT_MORE_EVIDENCE"
NO_LLM_LABEL = "aucun (règles déterministes)"

AnalyzerFactory = Callable[[Toolbox | None], SupplementalAnalyzer]


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
    llm_answered: bool | None = None
    llm_failure: str | None = None
    llm_retries: int | None = None


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


def _llm_failure(result: DiagnosisAgentResult) -> str | None:
    return next(
        (
            item
            for item in result.diagnosis.limitations
            if item.startswith("Supplemental analysis failed")
        ),
        "no LLM answer",
    )


def evaluate_scenario(
    scenario: Scenario,
    mode: Mode,
    qdrant: QdrantKnowledgeRetriever | None,
    analyzer_factory: AnalyzerFactory | None = None,
) -> Outcome:
    toolbox = (
        static_toolbox(scenario.tool_responses) if mode.tools and scenario.tool_responses else None
    )
    if mode.llm and analyzer_factory is None:
        raise ValueError(f"mode {mode.name} needs an analyzer factory")
    analyzer = analyzer_factory(toolbox) if mode.llm and analyzer_factory is not None else None
    agent = DiagnosisAgent(
        retriever=qdrant if mode.qdrant else None,
        analyzer=analyzer,
        toolbox=toolbox,
    )
    expected = scenario.expected
    started = perf_counter()
    try:
        diagnosis = diagnose_agent1_event(scenario_event(scenario), agent)[0].result
    except Exception as exc:  # A failing scenario is reported, not fatal for the run.
        return Outcome(
            scenario.scenario_id,
            mode.name,
            "ERROR",
            expected.root_cause,
            False,
            False,
            False,
            "NONE",
            False,
            [],
            True,
            0.0,
            None,
            None,
            (perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = (perf_counter() - started) * 1000
    codes = [hypothesis.code for hypothesis in diagnosis.diagnosis.hypotheses]
    accepted = {expected.root_cause, *expected.acceptable_root_causes}
    actions = _actions(diagnosis)
    first_action = _first_action(actions)
    succeeded_calls = {
        f"{call.tool}.{call.operation}"
        for call in diagnosis.tool_calls
        if call.status == "succeeded"
    }
    llm_answered = None
    if mode.llm:
        supplement = diagnosis.supplemental_analysis
        llm_answered = supplement is not None and supplement.source == "llm"
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
        llm_answered=llm_answered,
        llm_failure=_llm_failure(diagnosis) if llm_answered is False else None,
        llm_retries=getattr(analyzer, "retries", None) if mode.llm else None,
    )


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    rag = [item.rag_hit for item in outcomes if item.rag_hit is not None]
    required = [item.required_tools_ok for item in outcomes if item.required_tools_ok is not None]
    answered = [item.llm_answered for item in outcomes if item.llm_answered is not None]
    return {
        "scenarios": len(outcomes),
        "errors": sum(item.error is not None for item in outcomes),
        "rca_accuracy_top1": _rate([item.top1_correct for item in outcomes]),
        "rca_accuracy_acceptable": _rate([item.acceptable_correct for item in outcomes]),
        "rca_accuracy_top3": _rate([item.top3_correct for item in outcomes]),
        "first_action_accuracy": _rate([item.action_correct for item in outcomes]),
        "unsafe_recommendation_rate": _rate([bool(item.unsafe_actions) for item in outcomes]),
        # A wrong cause presented without asking for more evidence: the error that leads to a
        # wrong automatic action. Flagging it (requires_additional_evidence) is the safe outcome.
        "confident_errors": sum(_confident_error(item) for item in outcomes),
        "needs_more_evidence_rate": _rate([item.needs_more_evidence for item in outcomes]),
        "mean_evidence_strength": round(mean(item.evidence_strength for item in outcomes), 3),
        "rag_hit_rate": _rate(rag),
        "requested_evidence_collected_rate": _rate(required),
        "mean_decision_latency_ms": round(mean(item.latency_ms for item in outcomes), 1),
        "llm_answer_rate": _rate(answered),
        "llm_retries": sum(item.llm_retries or 0 for item in outcomes) if answered else None,
        "llm_timeouts": sum("timed out" in (item.llm_failure or "") for item in outcomes)
        if answered
        else None,
    }


def _confident_error(item: Outcome | dict[str, Any]) -> bool:
    """A wrong cause presented without asking for more evidence (leads to a wrong action)."""
    data = asdict(item) if isinstance(item, Outcome) else item
    return not data["acceptable_correct"] and not data["needs_more_evidence"]


def avoided_confident_errors(baseline: list[dict[str, Any]], with_llm: list[dict[str, Any]]) -> int:
    """Scenarios where the rules erred confidently and the LLM review flagged or fixed it."""
    reviewed = {item["scenario_id"]: item for item in with_llm}
    return sum(
        _confident_error(item) and not _confident_error(reviewed[item["scenario_id"]])
        for item in baseline
        if item["scenario_id"] in reviewed
    )


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


def run_evaluation(
    scenarios: list[Scenario],
    analyzer_factory: AnalyzerFactory | None = None,
    model_label: str | None = None,
    scenario_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    qdrant_url: str | None = None,
    llm_runs: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every mode; ``full+llm`` is added when an analyzer factory is given.

    ``llm_runs`` repeats the LLM mode (its answers vary); ``llm_runs`` in the report keeps each
    run's metrics, and the main tables show the first run.

    ``scenario_delay_seconds`` pauses between LLM scenarios to stay under free-tier rate limits.
    ``qdrant_url`` uses an indexed Qdrant server instead of an in-memory store.
    """
    modes = (*MODES, LLM_MODE) if analyzer_factory is not None else MODES
    qdrant = (
        QdrantKnowledgeRetriever.connect(url=qdrant_url)
        if qdrant_url
        else QdrantKnowledgeRetriever.in_memory()
    )

    def run_mode(mode: Mode) -> list[Outcome]:
        results = []
        for index, item in enumerate(scenarios):
            if mode.llm and index and scenario_delay_seconds > 0:
                sleep(scenario_delay_seconds)
            if progress is not None:
                progress(
                    f"[{datetime.now():%H:%M:%S}] {mode.name} {index + 1}/{len(scenarios)} "
                    f"{item.scenario_id}"
                )
            results.append(evaluate_scenario(item, mode, qdrant, analyzer_factory))
        return results

    outcomes = {mode.name: run_mode(mode) for mode in modes}
    repeats = []
    if analyzer_factory is not None:
        # LLM answers vary between runs; the deterministic modes do not, so only the LLM repeats.
        baseline = [asdict(item) for item in outcomes[MODES[-1].name]]
        runs = [outcomes[LLM_MODE.name], *(run_mode(LLM_MODE) for _ in range(llm_runs - 1))]
        repeats = [
            {
                **summarize(run),
                "avoided_confident_errors": avoided_confident_errors(
                    baseline, [asdict(i) for i in run]
                ),
            }
            for run in runs
        ]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "modes": [mode.name for mode in modes],
        "models": {
            mode.name: (model_label or "LLM") if mode.llm else NO_LLM_LABEL for mode in modes
        },
        "scenario_delay_seconds": scenario_delay_seconds if analyzer_factory is not None else 0.0,
        "qdrant": qdrant_url or "in-memory",
        "metrics": {name: summarize(items) for name, items in outcomes.items()},
        "outcomes": {name: [asdict(item) for item in items] for name, items in outcomes.items()},
        "llm_runs": repeats,
        "reliability": duplicate_delivery_check(scenarios[0]),
        "scenarios": {item.scenario_id: item.title for item in scenarios},
    }


_METRIC_LABELS = {
    "rca_accuracy_top1": "Précision RCA (top-1)",
    "rca_accuracy_acceptable": "Précision RCA (causes acceptables)",
    "rca_accuracy_top3": "Bonne cause dans le top-3",
    "first_action_accuracy": "Première action correcte",
    "unsafe_recommendation_rate": "Recommandations dangereuses (↓)",
    "confident_errors": "Erreurs sûres d’elles : mauvaise cause sans demande de preuves (↓)",
    "needs_more_evidence_rate": "Demande de preuves supplémentaires",
    "mean_evidence_strength": "Force moyenne des preuves",
    "rag_hit_rate": "RAG : incident passé pertinent en tête",
    "requested_evidence_collected_rate": "Preuves demandées collectées (replan)",
    "mean_decision_latency_ms": "Latence moyenne de décision (ms)",
    "llm_answer_rate": "Réponses LLM utilisables",
    "llm_retries": "Nouvelles tentatives après limite de débit (429)",
    "llm_timeouts": "Délais dépassés (réponse LLM trop lente)",
    "errors": "Erreurs",
}
_RAW_METRICS = ("mean_decision_latency_ms", "errors", "llm_retries", "llm_timeouts")
_COUNT_METRICS = ("confident_errors",)


def _llm_runs_section(report: dict[str, Any]) -> list[str]:
    runs = report.get("llm_runs") or []
    if not runs:
        return []
    total = runs[0]["scenarios"]
    avoided = [run["avoided_confident_errors"] for run in runs]
    lines = [
        "",
        f"## Apport du LLM ({len(runs)} lancement{'s' if len(runs) > 1 else ''})",
        "",
        f"Comparé à `{MODES[-1].name}` (mêmes outils et même RAG, sans LLM).",
        "",
        "| Lancement | Bonne cause | 1re action | Erreurs sûres d’elles | Évitées par le LLM "
        "| Réponses LLM | Délais dépassés | 429 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, run in enumerate(runs, 1):
        lines.append(
            f"| {index} | {_cell(run['rca_accuracy_acceptable'])} | "
            f"{_cell(run['first_action_accuracy'])} | "
            f"{run['confident_errors']} sur {total} | {run['avoided_confident_errors']} sur "
            f"{total} | "
            f"{_cell(run['llm_answer_rate'])} | {run['llm_timeouts']} | {run['llm_retries']} |"
        )
    if len(runs) > 1:

        def stats(key: str) -> tuple[float, Any]:
            values = [run[key] for run in runs]
            return mean(values), values

        accuracy, accuracies = stats("rca_accuracy_acceptable")
        action, actions = stats("first_action_accuracy")
        answers, answer_rates = stats("llm_answer_rate")
        errors, error_counts = stats("confident_errors")
        lines += [
            f"| **Moyenne** | {accuracy:.0%} | {action:.0%} | {errors:.1f} sur {total} | "
            f"{mean(avoided):.1f} sur {total} | {answers:.0%} | "
            f"{sum(run['llm_timeouts'] for run in runs)} au total | "
            f"{sum(run['llm_retries'] for run in runs)} au total |",
            f"| **Pire cas** | {min(accuracies):.0%} | {min(actions):.0%} | "
            f"{max(error_counts)} sur {total} | "
            f"{min(avoided)} sur {total} | {min(answer_rates):.0%} | "
            f"{max(run['llm_timeouts'] for run in runs)} | "
            f"{max(run['llm_retries'] for run in runs)} |",
        ]
    lines += [
        "",
        f"En résumé : {min(avoided)} à {max(avoided)} erreur(s) sûre(s) d’elle(s) évitée(s) "
        f"sur {total} "
        "selon le lancement. Le LLM ne choisit que parmi les causes des règles : il corrige un "
        "classement serré ou signale un désaccord, sans jamais déclencher d’action.",
    ]
    return lines


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
        "## Configurations et modèles",
        "",
        "| Configuration | Modèle utilisé |",
        "|---|---|",
        *(f"| `{mode}` | {report.get('models', {}).get(mode, NO_LLM_LABEL)} |" for mode in modes),
    ]
    lines.append(f"\nQdrant : {report.get('qdrant', 'in-memory')}.")
    if report.get("scenario_delay_seconds"):
        lines.append(f"\nDélai entre les scénarios LLM : {report['scenario_delay_seconds']} s.")
    lines += [
        "",
        "## Métriques par configuration",
        "",
        "| Métrique | " + " | ".join(f"`{mode}`" for mode in modes) + " |",
        "|---|" + "---:|" * len(modes),
    ]
    for key, label in _METRIC_LABELS.items():
        values = []
        for mode in modes:
            value = report["metrics"][mode].get(key)
            if key in _COUNT_METRICS and value is not None:
                values.append(f"{value} sur {report['metrics'][mode]['scenarios']}")
            else:
                values.append(
                    _cell(value) if value is None or key not in _RAW_METRICS else f"{value}"
                )
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(_llm_runs_section(report))
    lines.extend(["", "## Résultat par scénario", ""])
    lines.append("| Scénario | Attendu | " + " | ".join(f"`{mode}`" for mode in modes) + " |")
    lines.append("|---|---|" + "---|" * len(modes))
    by_mode = {
        mode: {item["scenario_id"]: item for item in report["outcomes"][mode]} for mode in modes
    }
    for scenario_id in report["scenarios"]:
        expected = by_mode[modes[0]][scenario_id]["expected"]
        cells = []
        for mode in modes:
            item = by_mode[mode][scenario_id]
            mark = "✅" if item["acceptable_correct"] else "❌"
            unsafe = " ⚠️ " + ", ".join(item["unsafe_actions"]) if item["unsafe_actions"] else ""
            fallback = " (LLM non utilisé)" if item.get("llm_answered") is False else ""
            cells.append(f"{mark} {item['predicted']} → {item['first_action']}{unsafe}{fallback}")
        lines.append(f"| {scenario_id} | {expected} | " + " | ".join(cells) + " |")
    failures = [
        f"- {mode} / {item['scenario_id']} : {item['llm_failure']}"
        for mode in modes
        for item in report["outcomes"][mode]
        if item.get("llm_failure")
    ]
    if failures:
        lines.extend(["", "## Réponses LLM non utilisées (repli sur les règles)", "", *failures])
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
        nargs="?",
        const="openai",
        choices=("openai", "anthropic"),
        help="Also evaluate with an LLM. openai (default): OpenAI-compatible endpoint from "
        "LLM_BASE_URL, LLM_MODEL, LLM_API_KEY_ENV; anthropic: Claude (billed).",
    )
    parser.add_argument(
        "--scenario-delay",
        type=float,
        default=float(os.environ.get("LLM_SCENARIO_DELAY_SECONDS") or 0),
        help="Seconds to wait between LLM scenarios (default: LLM_SCENARIO_DELAY_SECONDS or 0).",
    )
    parser.add_argument(
        "--qdrant-url", help="Indexed Qdrant server for the qdrant modes (default: in-memory)."
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Run the LLM mode this many times (answers vary)."
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    scenarios = load_scenarios(args.scenarios_dir)
    if not scenarios:
        print(f"No scenarios found in {args.scenarios_dir}", file=sys.stderr)
        return 2
    factory: AnalyzerFactory | None = None
    model_label = None
    if args.llm:
        try:
            # Built once up front so a configuration error stops the run before any call.
            analyzer = build_analyzer_from_env(provider=args.llm)
        except LLMConfigurationError as exc:
            print(f"LLM configuration error: {exc}", file=sys.stderr)
            return 2
        if analyzer is None:
            print(f"LLM configuration error: provider '{args.llm}' is off", file=sys.stderr)
            return 2
        model_label = analyzer.model_label

        def factory(toolbox: Toolbox | None) -> SupplementalAnalyzer:
            built = build_analyzer_from_env(toolbox, provider=args.llm)
            assert built is not None
            return built

        print(f"LLM: {model_label}")
    report = run_evaluation(
        scenarios,
        factory,
        model_label,
        args.scenario_delay,
        qdrant_url=args.qdrant_url,
        llm_runs=args.repeat,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
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
            + (
                f"  LLM answers {_cell(metrics['llm_answer_rate'])}"
                if metrics["llm_answer_rate"] is not None
                else ""
            )
        )
    print(f"Report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
