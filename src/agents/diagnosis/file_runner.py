"""Run Agent 2 locally from a user-supplied incident JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
)

from src.agents.diagnosis.contracts import (
    DiagnosisAgentResult,
    DiagnosisContract,
    DiagnosisInput,
    EvidenceItem,
    RecentChangeEvent,
)
from src.agents.diagnosis.service import DiagnosisAgent
from src.contracts.enums import AgentId, Environment, ResultStatus
from src.contracts.tasks import AgentTask, TaskContext


class IncidentFile(DiagnosisContract):
    """Local JSON boundary; evidence content uses the existing Agent 2 evidence contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(..., min_length=1)
    service_id: str = Field(..., min_length=1)
    namespace: str = Field(..., min_length=1)
    environment: Environment
    issue_summary: str = Field(..., min_length=1)
    signals: dict[str, JsonValue] = Field(default_factory=dict)
    recent_changes: list[str] = Field(default_factory=list)
    recent_change_events: list[RecentChangeEvent] = Field(default_factory=list)
    memory_rise_started_at: AwareDatetime | None = None
    memory_growth_observed: bool = False
    evidence_refs: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    evidence: list[EvidenceItem] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    context: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("evidence_refs")
    @classmethod
    def validate_unique_evidence_refs(cls, evidence_refs: tuple[str, ...]) -> tuple[str, ...]:
        if len(evidence_refs) != len(set(evidence_refs)):
            raise ValueError("Evidence references must be unique")
        return evidence_refs

    def to_diagnosis_input(self) -> DiagnosisInput:
        issue_summary = self.issue_summary
        if self.context:
            context_json = json.dumps(self.context, ensure_ascii=False, sort_keys=True)
            issue_summary = f"{issue_summary}\nAdditional incident context: {context_json}"
        return DiagnosisInput(
            incident_id=self.incident_id,
            service_id=self.service_id,
            namespace=self.namespace,
            environment=self.environment,
            issue_summary=issue_summary,
            signals=dict(self.signals),
            recent_changes=list(self.recent_changes),
            recent_change_events=list(self.recent_change_events),
            memory_rise_started_at=self.memory_rise_started_at,
            memory_growth_observed=self.memory_growth_observed,
            evidence=list(self.evidence),
            constraints=list(self.constraints),
        )


class DiagnosisFileError(ValueError):
    """A user-facing incident file read or validation error."""


def _validation_message(exc: ValidationError) -> str:
    details = []
    for error in exc.errors(include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        details.append(f"{location}: {error['msg']}")
    return "; ".join(details)


def load_incident_file(path: Path) -> IncidentFile:
    """Read and validate one local incident document without modifying it."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosisFileError(f"Cannot read incident file '{path}': {exc}") from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DiagnosisFileError(
            f"Invalid JSON in '{path}' at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    try:
        return IncidentFile.model_validate(document)
    except ValidationError as exc:
        raise DiagnosisFileError(
            f"Invalid incident fields in '{path}': {_validation_message(exc)}"
        ) from exc


def render_markdown_report(result: DiagnosisAgentResult, incident: IncidentFile) -> str:
    """Render the structured result as an evidence-linked human-readable report."""
    diagnosis = result.diagnosis
    lines = [
        f"# Diagnostic de l’incident {result.incident_id}",
        "",
        f"- **Service :** {incident.service_id}",
        f"- **Namespace :** {incident.namespace}",
        f"- **Environnement :** {incident.environment.value}",
        f"- **Description :** {incident.issue_summary}",
        f"- **Contexte fourni :** "
        f"`{json.dumps(incident.context, ensure_ascii=False, sort_keys=True)}`"
        if incident.context
        else "- **Contexte fourni :** aucun",
        f"- **Statut :** {result.status.value}",
        f"- **Diagnostic :** {diagnosis.diagnosis_id}",
        f"- **Cause principale :** `{diagnosis.primary_root_cause}`",
        f"- **Preuves supplémentaires nécessaires :** "
        f"{'oui' if diagnosis.requires_additional_evidence else 'non'}",
        f"- **Force des preuves (0-1) :** {result.evidence_strength:.2f}",
        "",
        "## Résumé",
        "",
        diagnosis.rationale_summary,
        "",
        "## Hypothèses classées",
        "",
        "| Rang | Hypothèse | Score heuristique | Références de preuves |",
        "|---:|---|---:|---|",
    ]
    for hypothesis in diagnosis.hypotheses:
        references = (
            ", ".join(f"`{reference}`" for reference in hypothesis.evidence_refs) or "Aucune"
        )
        lines.append(
            f"| {hypothesis.rank} | `{hypothesis.code}` | "
            f"{hypothesis.probability:.0%} | {references} |"
        )
    lines.extend(
        [
            "",
            "> Les scores sont des classements heuristiques, pas des probabilités calibrées.",
            "",
            "## Explicabilité des hypothèses",
            "",
        ]
    )
    for explanation in result.explainability.hypothesis_explanations:
        lines.extend(
            [
                f"### Rang {explanation.rank} — `{explanation.code}`",
                "",
                explanation.explanation,
                "",
                f"- **Classement :** {explanation.ranking_reason}",
                "- **Éléments favorables :** "
                + (
                    ", ".join(f"`{ref}`" for ref in explanation.supporting_evidence_refs)
                    or "aucun identifié"
                ),
                "- **Éléments défavorables :** "
                + (
                    ", ".join(f"`{ref}`" for ref in explanation.opposing_evidence_refs)
                    or "aucun observé; absence de données non interprétée comme preuve contraire"
                ),
            ]
        )
        if explanation.missing_information:
            lines.extend(
                [
                    "- **Informations manquantes :**",
                    *[f"  - {item}" for item in explanation.missing_information],
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Provenance des données",
            "",
            "- **Faits / preuves fournis :** "
            + (
                ", ".join(f"`{ref}`" for ref in result.explainability.observed_evidence_refs)
                or "aucun"
            ),
            "- **Preuves dérivées des entrées :** "
            + (
                ", ".join(f"`{ref}`" for ref in result.explainability.derived_evidence_refs)
                or "aucune"
            ),
            "- **Contexte de fixtures :** "
            + (
                ", ".join(f"`{ref}`" for ref in result.explainability.fixture_context_refs)
                or "aucun"
            ),
            "",
            "## Contradictions et données manquantes",
            "",
        ]
    )
    if result.explainability.contradictions:
        lines.extend(
            f"- **Contradiction :** {item}" for item in result.explainability.contradictions
        )
    else:
        lines.append("Aucune contradiction de seuil identifiée.")
    if result.explainability.missing_information:
        lines.extend(
            f"- **Manquant :** {item}" for item in result.explainability.missing_information
        )
    else:
        lines.append("Aucune information manquante signalée par l’analyse.")
    lines.extend(
        [
            "",
            "## Preuves disponibles",
            "",
        ]
    )
    if result.evidence:
        for item in result.evidence:
            observed_at = item.observed_at.isoformat() if item.observed_at else "non fourni"
            lines.extend(
                [
                    f"### `{item.evidence_id}` — {item.evidence_type}",
                    "",
                    f"- **Source / provenance :** {item.source} / {item.provenance}",
                    f"- **Horodatage :** {observed_at}",
                    f"- **Résumé :** {item.summary}",
                    "- **Valeurs :** "
                    f"`{json.dumps(item.value, ensure_ascii=False, sort_keys=True)}`",
                    "",
                ]
            )
    else:
        lines.extend(["Aucune preuve exploitable fournie ou dérivée.", ""])

    lines.extend(["## Recommandations", ""])
    if diagnosis.recommended_actions:
        for action in diagnosis.recommended_actions:
            lines.append(
                f"- **{action.action.value}** pour `{action.target.service}` "
                f"dans `{action.target.namespace}` — paramètres : "
                f"`{json.dumps(action.parameters, ensure_ascii=False, sort_keys=True)}`."
            )
    else:
        lines.append("Aucune action partagée proposée.")
    if result.explainability.recommendation_explanations:
        lines.append("")
        for justification in result.explainability.recommendation_explanations:
            lines.append(
                f"- **Justification de {justification.action} :** {justification.justification}"
            )
            lines.append(
                "- **Preuves liées :** "
                + (", ".join(f"`{ref}`" for ref in justification.evidence_refs) or "aucune")
            )
            lines.extend(f"- **Limite :** {item}" for item in justification.limitations)

    lines.extend(["", "## Connaissances récupérées (RAG local, pas une recherche Qdrant)", ""])
    knowledge = result.knowledge
    if knowledge.similar_incidents or knowledge.runbooks:
        if knowledge.similar_incidents:
            lines.extend(
                [
                    "### Incidents passés similaires",
                    "",
                    "| Référence | Cause passée | Score | Résumé |",
                    "|---|---|---:|---|",
                ]
            )
            lines.extend(
                f"| `{item.reference_id}` | `{item.root_cause}` | {item.score:.2f} | "
                f"{item.summary} |"
                for item in knowledge.similar_incidents
            )
            lines.append("")
        if knowledge.runbooks:
            lines.extend(["### Runbooks applicables", ""])
            lines.extend(
                f"- `{item.reference_id}` (score {item.score:.2f}) : {item.summary}"
                for item in knowledge.runbooks
            )
            lines.append("")
        lines.extend(
            [
                "> Ces documents servent de contexte : ils ne sont jamais utilisés comme preuves "
                "de l’incident courant.",
                "",
            ]
        )
    elif result.knowledge_refs:
        lines.extend(
            [
                "Références de connaissances : "
                + ", ".join(f"`{reference}`" for reference in result.knowledge_refs),
                "",
            ]
        )
    else:
        lines.extend(["Aucune référence de connaissance récupérée.", ""])

    lines.extend(["## Investigation (outils en lecture seule)", ""])
    if result.tool_calls:
        lines.extend(
            [
                "| Outil | Opération | Demandé par | Statut | Preuves / erreur |",
                "|---|---|---|---|---|",
            ]
        )
        for call in result.tool_calls:
            detail = ", ".join(f"`{ref}`" for ref in call.evidence_refs) or (call.error or "aucune")
            lines.append(
                f"| {call.tool} | {call.operation} | {call.requested_by} | {call.status} | "
                f"{detail} |"
            )
        lines.append("")
    else:
        lines.extend(["Aucun outil de collecte configuré pour ce diagnostic.", ""])

    analysis = result.supplemental_analysis
    lines.extend(["## Analyse complémentaire", ""])
    if analysis is not None:
        origin = "modèle de langage" if analysis.source == "llm" else "simulation déterministe"
        lines.extend(
            [
                f"- **Source :** {origin}",
                f"- **Cause retenue :** `{analysis.selected_cause or 'aucune'}`",
                f"- **Résumé :** {analysis.summary}",
                "",
            ]
        )
    else:
        lines.extend(["Indisponible ou rejetée à la validation; voir les limites.", ""])

    lines.extend(["## Limites et incertitudes", ""])
    if diagnosis.limitations:
        lines.extend(f"- {limitation}" for limitation in diagnosis.limitations)
    else:
        lines.append("- Aucune limite signalée par le pipeline.")
    lines.extend(
        [
            "",
            "Ce rapport provient du pipeline local déterministe. Il n’exécute aucune action "
            "d’infrastructure; le RAG est une recherche lexicale locale et aucun LLM réel "
            "n’est appelé.",
            "",
        ]
    )
    return "\n".join(lines)


def run_incident_file(
    input_path: Path,
    output_dir: Path | None = None,
) -> tuple[DiagnosisAgentResult, Path, Path]:
    """Run the local diagnosis and write structured JSON plus a Markdown report."""
    incident = load_incident_file(input_path)
    payload = incident.to_diagnosis_input()
    task_context = TaskContext(
        service_id=incident.service_id,
        namespace=incident.namespace,
        environment=incident.environment,
        signals=incident.signals,
        evidence_refs=incident.evidence_refs,
        constraints=tuple(incident.constraints),
    )
    task = AgentTask(
        task_id=f"FILE-{incident.incident_id}",
        incident_id=incident.incident_id,
        agent_id=AgentId.DIAGNOSIS,
        objective=incident.issue_summary,
        task_context=task_context,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )

    result = DiagnosisAgent().diagnose_input(task, payload)
    if result.status is not ResultStatus.SUCCEEDED or not isinstance(result, DiagnosisAgentResult):
        raise DiagnosisFileError(
            f"Diagnosis failed for '{incident.incident_id}': {result.rationale_summary}"
        )

    target_dir = output_dir or input_path.parent
    json_path = target_dir / f"{input_path.stem}.diagnosis.json"
    markdown_path = target_dir / f"{input_path.stem}.diagnosis.md"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown_report(result, incident), encoding="utf-8")
    except OSError as exc:
        raise DiagnosisFileError(f"Cannot write diagnosis report to '{target_dir}': {exc}") from exc
    return result, json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent 2 on a local incident JSON file.")
    parser.add_argument("incident_file", type=Path, help="Input incident JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the JSON result and Markdown report (defaults to input directory).",
    )
    args = parser.parse_args(argv)

    try:
        result, json_path, markdown_path = run_incident_file(args.incident_file, args.output_dir)
    except DiagnosisFileError as exc:
        print(f"Diagnosis input/output error: {exc}", file=sys.stderr)
        return 2

    print(f"Diagnosis {result.incident_id}: {result.diagnosis.primary_root_cause}")
    print(f"JSON result: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
