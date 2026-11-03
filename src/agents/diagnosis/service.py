from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import ValidationError

from src.agents.base import SkeletonAgent
from src.agents.diagnosis.contracts import (
    DiagnosisAgentResult,
    DiagnosisInput,
    EvidenceItem,
)
from src.agents.diagnosis.evidence import ResolvedEvidence, validate_resolved_evidence_scope
from src.agents.diagnosis.ports import KnowledgeRetriever, SupplementalAnalyzer
from src.agents.diagnosis.result_builder import build_result, to_diagnosis_completed
from src.agents.diagnosis.signal_adapter import adapt_agent1_signals
from src.agents.diagnosis.tools import Toolbox
from src.contracts.enums import AgentId, ResultStatus
from src.contracts.tasks import AgentResult, AgentTask


class DiagnosisAgent(SkeletonAgent):
    agent_id = AgentId.DIAGNOSIS
    responsibility = "Rank root causes and recommend evidence-backed actions"

    def __init__(
        self,
        retriever: KnowledgeRetriever | None = None,
        analyzer: SupplementalAnalyzer | None = None,
        toolbox: Toolbox | None = None,
    ) -> None:
        self._retriever = retriever
        self._analyzer = analyzer
        self._toolbox = toolbox

    async def handle(
        self,
        task: AgentTask,
        resolved_evidence: Sequence[ResolvedEvidence] = (),
    ) -> AgentResult:
        if task.agent_id != self.agent_id:
            raise ValueError("Task addressed to a different agent")
        if task.deadline <= datetime.now(UTC):
            raise ValueError("Task deadline has expired")
        validate_resolved_evidence_scope(task, resolved_evidence)

        task_context = task.task_context
        raw_changes = task_context.signals.get("recent_changes", [])
        if not isinstance(raw_changes, list):
            return self._failed_result(
                task,
                "Invalid diagnosis input: recent_changes must be a list of strings.",
            )
        recent_changes: list[str] = []
        for change in raw_changes:
            if not isinstance(change, str):
                return self._failed_result(
                    task,
                    "Invalid diagnosis input: recent_changes must be a list of strings.",
                )
            recent_changes.append(change)

        required_evidence = task_context.signals.get("required_evidence", [])
        if not isinstance(required_evidence, list) or not all(
            isinstance(name, str) for name in required_evidence
        ):
            return self._failed_result(
                task,
                "Invalid diagnosis input: required_evidence must be a list of strings.",
            )

        try:
            change_events = task_context.signals.get("recent_change_events", [])
            memory_rise_started_at = task_context.signals.get("memory_rise_started_at")
            memory_growth_observed = task_context.signals.get("memory_growth_observed", False)
            payload = DiagnosisInput.model_validate(
                {
                    "incident_id": task.incident_id,
                    "service_id": task_context.service_id,
                    "namespace": task_context.namespace,
                    "environment": task_context.environment,
                    "issue_summary": task.objective,
                    "signals": {
                        key: value
                        for key, value in task_context.signals.items()
                        if key
                        not in {
                            "recent_changes",
                            "recent_change_events",
                            "memory_rise_started_at",
                            "memory_growth_observed",
                            "required_evidence",
                        }
                    },
                    "recent_changes": recent_changes,
                    "recent_change_events": change_events,
                    "memory_rise_started_at": memory_rise_started_at,
                    "memory_growth_observed": memory_growth_observed,
                    "evidence": [self._to_evidence_item(item) for item in resolved_evidence],
                    "constraints": list(task_context.constraints),
                    "required_evidence": required_evidence,
                }
            )
        except ValidationError as exc:
            invalid_fields = sorted(
                {
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors(include_input=False)
                }
            )
            return self._failed_result(
                task,
                "Invalid diagnosis input or output fields: " + ", ".join(invalid_fields),
            )
        return self.diagnose_input(task, payload)

    def diagnose_input(self, task: AgentTask, payload: DiagnosisInput) -> AgentResult:
        """Run the shared local pipeline for an already validated diagnosis input."""
        if task.agent_id != self.agent_id:
            raise ValueError("Task addressed to a different agent")
        if task.deadline <= datetime.now(UTC):
            raise ValueError("Task deadline has expired")

        try:
            adaptation = adapt_agent1_signals(payload)
            adapted_payload = adaptation.payload
            result = build_result(
                adapted_payload,
                self._retriever,
                self._analyzer,
                self._toolbox,
                task.deadline,
            )
            diagnosis = to_diagnosis_completed(task, result)
            if adaptation.limitations:
                diagnosis = diagnosis.model_copy(
                    update={
                        "limitations": tuple(
                            dict.fromkeys((*diagnosis.limitations, *adaptation.limitations))
                        ),
                        "requires_additional_evidence": (
                            diagnosis.requires_additional_evidence or adaptation.degraded
                        ),
                    }
                )
        except ValidationError as exc:
            invalid_fields = sorted(
                {
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors(include_input=False)
                }
            )
            return self._failed_result(
                task,
                "Invalid diagnosis input or output fields: " + ", ".join(invalid_fields),
            )
        except ValueError as exc:
            return self._failed_result(task, f"Diagnosis validation failed: {exc}")

        resolved_ids = {item.evidence_id for item in adapted_payload.evidence}
        unresolved_refs = tuple(
            evidence_ref
            for evidence_ref in task.task_context.evidence_refs
            if evidence_ref not in resolved_ids
        )
        if unresolved_refs:
            missing_reference_note = (
                "Evidence references were not resolved and were not used: "
                + ", ".join(unresolved_refs)
                + "."
            )
            diagnosis = diagnosis.model_copy(
                update={"limitations": (*diagnosis.limitations, missing_reference_note)}
            )
            explainability = result.explainability.model_copy(
                update={
                    "missing_information": (
                        *result.explainability.missing_information,
                        "Unresolved evidence references were not available: "
                        + ", ".join(unresolved_refs)
                        + ".",
                    )
                }
            )
        else:
            explainability = result.explainability

        evidence_refs = tuple(
            dict.fromkeys(
                evidence_ref
                for hypothesis in diagnosis.hypotheses
                for evidence_ref in hypothesis.evidence_refs
            )
        )
        return DiagnosisAgentResult(
            result_id=f"RESULT-{task.task_id}",
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=ResultStatus.SUCCEEDED,
            rationale_summary=diagnosis.rationale_summary,
            evidence_refs=evidence_refs,
            # Heuristic ranking score of the leading hypothesis; not a calibrated probability.
            confidence=diagnosis.hypotheses[0].probability,
            diagnosis=diagnosis,
            evidence=result.evidence,
            knowledge_refs=result.knowledge_refs,
            knowledge=result.knowledge,
            explainability=explainability,
            evidence_strength=result.evidence_strength,
            tool_calls=result.tool_calls,
            supplemental_analysis=result.supplemental_analysis,
        )

    def _failed_result(self, task: AgentTask, reason: str) -> AgentResult:
        return AgentResult(
            result_id=f"RESULT-{task.task_id}",
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=ResultStatus.FAILED,
            rationale_summary=reason,
        )

    @staticmethod
    def _to_evidence_item(item: ResolvedEvidence) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=item.evidence_id,
            evidence_type=item.evidence_type,
            source=item.source,
            provenance="provided",
            observed_at=item.observed_at,
            summary=item.summary,
            value=item.value,
        )
