from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.contracts.enums import Environment
from src.contracts.payloads import DiagnosisCompleted
from src.contracts.tasks import AgentResult


class DiagnosisContract(BaseModel):
    """Shared contract rules for the Agent 2 diagnostic layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)


EvidenceType = Literal["metric", "log", "k8s_event", "git_change", "config", "trace"]
EvidenceProvenance = Literal["provided", "derived", "fixture"]
KnowledgeKind = Literal["incident", "runbook"]


class EvidenceItem(DiagnosisContract):
    """A single, traceable observation used to justify a diagnosis hypothesis."""

    evidence_id: str = Field(..., min_length=1)
    evidence_type: EvidenceType
    source: str = Field(..., min_length=1)
    provenance: EvidenceProvenance = "provided"
    observed_at: AwareDatetime | None = None
    summary: str = Field(..., min_length=1)
    value: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RecentChangeEvent(DiagnosisContract):
    """Timestamped change record used only for temporal RCA qualification."""

    change_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    changed_at: AwareDatetime


class DiagnosisInput(DiagnosisContract):
    """Input payload supplied to the diagnosis agent by the orchestrator or test harness."""

    incident_id: str = Field(..., min_length=1)
    service_id: str = Field(..., min_length=1)
    namespace: str = Field(..., min_length=1)
    environment: Environment
    issue_summary: str = Field(..., min_length=1)
    signals: dict[str, Any] = Field(default_factory=dict)
    recent_changes: list[str] = Field(default_factory=list)
    recent_change_events: list[RecentChangeEvent] = Field(default_factory=list)
    memory_rise_started_at: AwareDatetime | None = None
    memory_growth_observed: bool = False
    evidence: list[EvidenceItem] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    # Evidence names requested by a replan (e.g. "heap_usage_breakdown", "gc_metrics").
    required_evidence: list[str] = Field(default_factory=list)

    @field_validator("signals")
    @classmethod
    def validate_known_signals(cls, signals: dict[str, Any]) -> dict[str, Any]:
        bounds = {
            "memory_percent": (0.0, 100.0),
            "memory_pct": (0.0, 100.0),
            "cpu_percent": (0.0, 100.0),
            "restart_count": (0.0, None),
            "latency_ms": (0.0, None),
            "error_rate": (0.0, 100.0),
        }
        for name, (minimum, maximum) in bounds.items():
            if name not in signals:
                continue
            value = signals[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Signal '{name}' must be a number")
            if not isfinite(value):
                raise ValueError(f"Signal '{name}' must be finite")
            if value < minimum or (maximum is not None and value > maximum):
                upper = f" and at most {maximum:g}" if maximum is not None else ""
                raise ValueError(f"Signal '{name}' must be at least {minimum:g}{upper}")
        return signals

    @field_validator("recent_changes")
    @classmethod
    def validate_recent_changes(cls, changes: list[str]) -> list[str]:
        cleaned = [change.strip() for change in changes]
        if any(not change for change in cleaned):
            raise ValueError("Recent changes cannot contain empty values")
        return cleaned

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "DiagnosisInput":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence identifiers must be unique within a diagnosis input")
        return self


class Hypothesis(DiagnosisContract):
    """One ranked explanation; probability is an uncalibrated heuristic score."""

    rank: int = Field(..., ge=1)
    code: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...] = Field(...)


class DiagnosisRecommendation(DiagnosisContract):
    """An action proposal; score is heuristic and does not authorize execution."""

    action: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(..., ge=0.0, le=1.0)


class KnowledgeDocument(DiagnosisContract):
    """Retrieved local knowledge; it is context, not evidence about the active incident."""

    reference_id: str = Field(..., min_length=1)
    kind: KnowledgeKind
    summary: str = Field(..., min_length=1)
    root_cause: str | None = None
    score: float = Field(..., ge=0.0, le=1.0)
    provenance: Literal["local_fixture", "qdrant", "curated_repository"] = "local_fixture"


class KnowledgeRetrieval(DiagnosisContract):
    """Separate prior incidents and runbooks returned by a knowledge retriever."""

    similar_incidents: tuple[KnowledgeDocument, ...] = ()
    runbooks: tuple[KnowledgeDocument, ...] = ()


class ToolCallRecord(DiagnosisContract):
    """Audit record of one read-only evidence tool call."""

    tool: str = Field(..., min_length=1)
    operation: str = Field(..., min_length=1)
    requested_by: Literal["investigation_plan", "llm"]
    status: Literal["succeeded", "failed", "skipped"]
    evidence_refs: tuple[str, ...] = ()
    error: str | None = None


class SupplementalAnalysis(DiagnosisContract):
    """Validated, non-authoritative supplemental analysis."""

    source: Literal["deterministic_simulation", "llm"]
    summary: str = Field(..., min_length=1)
    selected_cause: str | None = None
    evidence_refs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    # Evidence the analyzer gathered through read-only tools; rules re-evaluate it.
    collected_evidence: tuple[EvidenceItem, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()


class HypothesisExplanation(DiagnosisContract):
    """Human-readable basis and limitations for one ranked hypothesis."""

    rank: int = Field(..., ge=1)
    code: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    supporting_evidence_refs: tuple[str, ...] = ()
    opposing_evidence_refs: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    ranking_reason: str = Field(..., min_length=1)


class RecommendationExplanation(DiagnosisContract):
    """Evidence references and caveats for one recommendation."""

    action: str = Field(..., min_length=1)
    justification: str = Field(..., min_length=1)
    evidence_refs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class DiagnosisExplainability(DiagnosisContract):
    """Local-only explanation metadata; not part of the shared DiagnosisCompleted contract."""

    observed_evidence_refs: tuple[str, ...] = ()
    derived_evidence_refs: tuple[str, ...] = ()
    fixture_context_refs: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    hypothesis_explanations: tuple[HypothesisExplanation, ...] = ()
    recommendation_explanations: tuple[RecommendationExplanation, ...] = ()


class DiagnosisResult(DiagnosisContract):
    """Structured diagnosis result returned by the RCA agent."""

    diagnosis_id: str = Field(..., min_length=1)
    incident_id: str = Field(..., min_length=1)
    primary_root_cause: str = Field(..., min_length=1)
    hypotheses: list[Hypothesis] = Field(..., min_length=1)
    recommended_actions: list[DiagnosisRecommendation] = Field(default_factory=list)
    rationale_summary: str = Field(..., min_length=1)
    limitations: list[str] = Field(default_factory=list)
    requires_additional_evidence: bool = False
    evidence: tuple[EvidenceItem, ...] = ()
    knowledge_refs: tuple[str, ...] = ()
    knowledge: KnowledgeRetrieval = Field(default_factory=KnowledgeRetrieval)
    explainability: DiagnosisExplainability = Field(default_factory=DiagnosisExplainability)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_calls: tuple[ToolCallRecord, ...] = ()
    supplemental_analysis: SupplementalAnalysis | None = None


class DiagnosisAgentResult(AgentResult):
    """Local agent response retaining the complete shared diagnosis payload."""

    diagnosis: DiagnosisCompleted
    evidence: tuple[EvidenceItem, ...] = ()
    knowledge_refs: tuple[str, ...] = ()
    knowledge: KnowledgeRetrieval = Field(default_factory=KnowledgeRetrieval)
    explainability: DiagnosisExplainability = Field(default_factory=DiagnosisExplainability)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_calls: tuple[ToolCallRecord, ...] = ()
    supplemental_analysis: SupplementalAnalysis | None = None
