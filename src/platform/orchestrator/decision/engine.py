"""Deterministic recommendation decisions with a bounded optional advisor."""

import asyncio
import logging
import re
from typing import Protocol

from src.contracts.enums import ActionType, AgentId
from src.contracts.events import EventEnvelope
from src.contracts.payloads import (
    AnomalyDetected,
    DiagnosisCompleted,
    ModelHealthAssessed,
    ReleaseAssessment,
    ResourceRecommendation,
)
from src.platform.orchestrator.decision.contracts import (
    AdvisorRequest,
    AdvisorStatus,
    AdvisorSuggestion,
    CandidateSummary,
    DecisionDisposition,
    DecisionPlan,
    HypothesisSummary,
)

logger = logging.getLogger(__name__)


class DecisionAdvisor(Protocol):
    async def suggest(self, request: AdvisorRequest) -> AdvisorSuggestion: ...


class DecisionEngine:
    """Map typed specialist outputs to safe next steps.

    The advisor may express a preference among *existing* candidate IDs.
    It cannot create tasks, remove required gates, authorize actions or close incidents.
    """

    def __init__(
        self,
        advisor: DecisionAdvisor | None = None,
        *,
        advisor_timeout_seconds: float = 3.0,
        min_diagnosis_confidence: float = 0.6,
        min_resource_confidence: float = 0.7,
    ) -> None:
        if advisor_timeout_seconds <= 0:
            raise ValueError("Advisor timeout must be positive")
        if not 0 <= min_diagnosis_confidence <= 1 or not 0 <= min_resource_confidence <= 1:
            raise ValueError("Decision confidence thresholds must be between 0 and 1")
        self._advisor = advisor
        self._advisor_timeout_seconds = advisor_timeout_seconds
        self._min_diagnosis_confidence = min_diagnosis_confidence
        self._min_resource_confidence = min_resource_confidence

    async def decide(
        self, incoming: EventEnvelope, history: tuple[DecisionPlan, ...] = ()
    ) -> DecisionPlan:
        event = EventEnvelope.model_validate(incoming.model_dump(mode="json"))
        if event.topic == "monitoring.anomaly.detected":
            plan = self._anomaly(event)
        elif event.topic == "diagnosis.completed":
            plan = self._diagnosis(event)
        elif event.topic == "resource.recommendation.created":
            plan = self._resource(event)
        elif event.topic == "mlops.model.health.assessed":
            plan = self._model_health(event)
        elif event.topic == "quality.release.assessed":
            plan = self._quality(event)
        else:
            raise ValueError("No recommendation decision rule for this event topic")
        prior_actions = {
            candidate.action_id: candidate.action_fingerprint
            for prior in history
            for candidate in prior.candidates
        }
        if any(
            candidate.action_id in prior_actions
            and candidate.action_fingerprint != prior_actions[candidate.action_id]
            for candidate in plan.candidates
        ):
            return self._base(
                event,
                DecisionDisposition.BLOCKED,
                "ACTION_IDENTITY_CONFLICT",
                candidates=plan.candidates,
            )
        if (
            self._advisor is None
            or plan.disposition != DecisionDisposition.ROUTE_ANALYSIS
            or len(plan.candidates) < 2
        ):
            return plan
        if len(plan.candidates) > 16 or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", candidate.action_id)
            for candidate in plan.candidates
        ):
            return plan
        diagnosis = DiagnosisCompleted.model_validate(event.payload)
        hypotheses = tuple(
            HypothesisSummary(code=item.code, probability=item.probability)
            for item in sorted(diagnosis.hypotheses, key=lambda item: item.rank)
            if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", item.code)
        )[:5]
        if not hypotheses:
            return plan
        request = AdvisorRequest(
            incident_id=plan.incident_id,
            event_id=plan.event_id,
            environment=event.environment,
            candidates=plan.candidates,
            hypotheses=hypotheses,
            evidence_count=len(event.evidence_refs),
        )
        try:
            response = await asyncio.wait_for(
                self._advisor.suggest(request), timeout=self._advisor_timeout_seconds
            )
            suggestion = AdvisorSuggestion.model_validate(response.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("Decision advisor unavailable: %s", type(exc).__name__)
            return plan.model_copy(update={"advisor_status": AdvisorStatus.UNAVAILABLE})
        if suggestion.candidate_action_id not in {
            candidate.action_id for candidate in plan.candidates
        }:
            return plan.model_copy(update={"advisor_status": AdvisorStatus.REJECTED})
        return DecisionPlan.model_validate(
            {
                **plan.model_dump(mode="json"),
                "advisor_status": AdvisorStatus.ACCEPTED,
                "advisor_preferred_candidate_id": suggestion.candidate_action_id,
            }
        )

    @staticmethod
    def _base(
        event: EventEnvelope,
        disposition: DecisionDisposition,
        reason: str,
        *,
        agents: tuple[AgentId, ...] = (),
        candidates: tuple[CandidateSummary, ...] = (),
    ) -> DecisionPlan:
        return DecisionPlan(
            incident_id=event.correlation_id,
            event_id=event.message_id,
            topic=event.topic,
            disposition=disposition,
            next_agents=agents,
            candidates=candidates,
            reason_codes=(reason,),
        )

    def _anomaly(self, event: EventEnvelope) -> DecisionPlan:
        anomaly = AnomalyDetected.model_validate(event.payload)
        agents = [AgentId.DIAGNOSIS]
        model_id = anomaly.signals.get("model_id")
        model_version = anomaly.signals.get("model_version")
        if (model_id is None) != (model_version is None):
            raise ValueError("Model anomaly requires both model ID and version")
        if model_id is not None and (
            not isinstance(model_id, str)
            or not model_id.strip()
            or not isinstance(model_version, str)
            or not model_version.strip()
        ):
            raise ValueError("Model identity must contain non-empty strings")
        if model_id is not None:
            agents.append(AgentId.MLOPS_LIFECYCLE)
        return self._base(
            event,
            DecisionDisposition.ROUTE_ANALYSIS,
            "ANOMALY_REQUIRES_ANALYSIS",
            agents=tuple(agents),
        )

    def _diagnosis(self, event: EventEnvelope) -> DecisionPlan:
        diagnosis = DiagnosisCompleted.model_validate(event.payload)
        candidates = tuple(
            CandidateSummary(
                action_id=action.action_id,
                action_type=action.action,
                action_fingerprint=action.fingerprint(),
            )
            for action in diagnosis.recommended_actions
        )
        if diagnosis.requires_additional_evidence:
            return self._base(
                event,
                DecisionDisposition.WAIT_FOR_EVIDENCE,
                "DIAGNOSIS_NEEDS_EVIDENCE",
                candidates=candidates,
            )
        if not candidates:
            return self._base(event, DecisionDisposition.REVIEW_REQUIRED, "NO_ACTION_CANDIDATES")
        if (
            not diagnosis.hypotheses
            or max(item.probability for item in diagnosis.hypotheses)
            < self._min_diagnosis_confidence
        ):
            return self._base(
                event,
                DecisionDisposition.REVIEW_REQUIRED,
                "LOW_DIAGNOSIS_CONFIDENCE",
                candidates=candidates,
            )
        agents = [AgentId.SECURITY_QUALITY]
        types = {candidate.action_type for candidate in candidates}
        if ActionType.SCALE_REPLICAS in types:
            agents.append(AgentId.RESOURCE_OPTIMIZATION)
        if ActionType.PROMOTE_MODEL in types:
            agents.append(AgentId.MLOPS_LIFECYCLE)
        return self._base(
            event,
            DecisionDisposition.ROUTE_ANALYSIS,
            "CANDIDATES_REQUIRE_ASSESSMENT",
            agents=tuple(agents),
            candidates=candidates,
        )

    def _resource(self, event: EventEnvelope) -> DecisionPlan:
        recommendation = ResourceRecommendation.model_validate(event.payload)
        action = recommendation.candidate_action
        candidates = (
            CandidateSummary(
                action_id=action.action_id,
                action_type=action.action,
                action_fingerprint=action.fingerprint(),
            ),
        )
        if action.action != ActionType.SCALE_REPLICAS:
            return self._base(
                event,
                DecisionDisposition.REVIEW_REQUIRED,
                "UNEXPECTED_RESOURCE_ACTION",
                candidates=candidates,
            )
        if recommendation.confidence < self._min_resource_confidence:
            return self._base(
                event,
                DecisionDisposition.REVIEW_REQUIRED,
                "LOW_RESOURCE_CONFIDENCE",
                candidates=candidates,
            )
        return self._base(
            event,
            DecisionDisposition.ROUTE_ANALYSIS,
            "RESOURCE_ACTION_REQUIRES_QUALITY_GATE",
            agents=(AgentId.SECURITY_QUALITY,),
            candidates=candidates,
        )

    def _model_health(self, event: EventEnvelope) -> DecisionPlan:
        assessment = ModelHealthAssessed.model_validate(event.payload)
        if assessment.decision == "COLLECT_LABELS":
            return self._base(event, DecisionDisposition.WAIT_FOR_EVIDENCE, "MODEL_LABELS_MISSING")
        if assessment.decision == "HEALTHY":
            return self._base(event, DecisionDisposition.NO_ACTION, "MODEL_HEALTHY")
        return self._base(
            event,
            DecisionDisposition.ROUTE_ANALYSIS,
            "MODEL_RETRAINING_EVALUATION_REQUESTED",
            agents=(AgentId.MLOPS_LIFECYCLE,),
        )

    def _quality(self, event: EventEnvelope) -> DecisionPlan:
        assessment = ReleaseAssessment.model_validate(event.payload)
        if assessment.decision == "BLOCK":
            return self._base(event, DecisionDisposition.BLOCKED, "QUALITY_GATE_BLOCKED")
        # The v1 assessment has no exact action fingerprint. It cannot authorize
        # a candidate action even when a release is marked ALLOW.
        return self._base(
            event, DecisionDisposition.REVIEW_REQUIRED, "QUALITY_RESULT_NOT_ACTION_BOUND"
        )
