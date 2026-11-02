from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite

from src.agents.diagnosis.contracts import (
    DiagnosisInput,
    DiagnosisExplainability,
    DiagnosisRecommendation,
    DiagnosisResult,
    EvidenceItem,
    Hypothesis,
    HypothesisExplanation,
    KnowledgeRetrieval,
    RecommendationExplanation,
    SupplementalAnalysis,
    ToolCallRecord,
)
from src.agents.diagnosis.evidence_collector import collect_evidence
from src.agents.diagnosis.investigation import change_events_from_evidence, investigate
from src.agents.diagnosis.llm_analyzer import DeterministicSupplementalAnalyzer
from src.agents.diagnosis.ports import KnowledgeRetriever, SupplementalAnalyzer
from src.agents.diagnosis.rag_retriever import LocalKnowledgeRetriever
from src.agents.diagnosis.tools import Toolbox
from src.agents.diagnosis.rules import (
    conflicting_metrics,
    qualifies_as_prior_deployment,
    evaluate_hypotheses,
    has_oom_observation,
    signal_conflicts,
    _suspected_dependency,
    _uncertain_deployment_correlation,
)
from src.contracts.actions import ActionIntent, ActionTarget
from src.contracts.enums import ActionType
from src.contracts.payloads import DiagnosisCompleted, Hypothesis
from src.contracts.tasks import AgentTask


DEFAULT_INVESTIGATION_TIMEOUT = timedelta(minutes=2)
# A language model may reorder rule candidates whose scores differ by at most this margin.
LLM_RERANK_MARGIN = 0.15
KNOWLEDGE_LIMIT = 3
# Codes that describe what is happening, not why: the cause still has to be established.
SYMPTOM_LEVEL_CAUSES = frozenset(
    {"INSUFFICIENT_EVIDENCE", "MEMORY_GROWTH", "MEMORY_PRESSURE", "CRASH_LOOP_UNEXPLAINED"}
)
RETRIEVAL_CANDIDATES = 8

# Past root causes that are not identical to a candidate but describe the same failure family.
RELATED_ROOT_CAUSES: dict[str, frozenset[str]] = {
    "MEMORY_LIMIT_REACHED": frozenset({"INSUFFICIENT_MEMORY_LIMIT", "MEMORY_LEAK_POST_DEPLOY"}),
    "MEMORY_GROWTH": frozenset({"MEMORY_LEAK_POST_DEPLOY", "INSUFFICIENT_MEMORY_LIMIT"}),
    "MEMORY_PRESSURE": frozenset({"MEMORY_LEAK_POST_DEPLOY", "INSUFFICIENT_MEMORY_LIMIT"}),
    "MEMORY_LEAK_POST_DEPLOY": frozenset({"INSUFFICIENT_MEMORY_LIMIT"}),
    "DEPENDENCY_UNAVAILABLE": frozenset({"CONNECTION_POOL_EXHAUSTION"}),
    "CONNECTION_POOL_EXHAUSTION": frozenset({"DEPENDENCY_UNAVAILABLE"}),
    "TRAFFIC_SPIKE": frozenset({"RESOURCE_PRESSURE_OR_CPU_STARVATION", "CONNECTION_POOL_EXHAUSTION"}),
    "RESOURCE_PRESSURE_OR_CPU_STARVATION": frozenset({"TRAFFIC_SPIKE"}),
    "CONFIG_ERROR_CRASHLOOP": frozenset({"CRASH_LOOP_UNEXPLAINED"}),
    "CRASH_LOOP_UNEXPLAINED": frozenset({"CONFIG_ERROR_CRASHLOOP"}),
}


_NON_SEARCH_SIGNAL_KEYS = frozenset(
    {
        "agent1_context", "categories", "observed_on", "incidents", "first_seen", "suspects",
        "decision_reason", "alert_id", "change_context", "detected_at", "baseline",
    }
)


def _agent1_categories(payload: DiagnosisInput) -> tuple[str, ...]:
    context = payload.signals.get("agent1_context")
    categories = context.get("categories") if isinstance(context, dict) else None
    if not isinstance(categories, list):
        return ()
    return tuple(category for category in categories if isinstance(category, str))


def _retrieval_query(
    payload: DiagnosisInput,
    evidence: list[EvidenceItem],
    hypotheses: list[Hypothesis] | None = None,
) -> str:
    # Measured signal names only; Agent 1 envelope bookkeeping keys add noise to the search.
    signal_names = " ".join(
        name for name in payload.signals if name not in _NON_SEARCH_SIGNAL_KEYS and "." not in name
    )
    evidence_summaries = " ".join(item.summary for item in evidence)
    # Underscored codes are split so they match corpus tags such as "memory" or "limit".
    codes = [
        hypothesis.code.replace("_", " ").casefold()
        for hypothesis in hypotheses or ()
        if hypothesis.code != "INSUFFICIENT_EVIDENCE"
    ]
    # The leading candidate is repeated so it weighs more than lower-ranked alternatives.
    candidate_words = " ".join([*codes[:1], *codes])
    category_words = " ".join(
        category.replace("_", " ") for category in _agent1_categories(payload)
    )
    return " ".join(
        part
        for part in (
            payload.issue_summary,
            signal_names,
            " ".join(payload.recent_changes),
            evidence_summaries,
            candidate_words,
            category_words,
        )
        if part
    )


def _root_cause_relation(document_cause: str | None, hypotheses: list[Hypothesis]) -> int:
    """Rank agreement of a past root cause with the candidates, leading candidate first.

    4 = same cause as the leading candidate, 3 = related to it, 2 = same cause as an
    alternative candidate, 1 = related to an alternative, 0 = unrelated.
    """
    if document_cause is None or not hypotheses:
        return 0
    leading, *alternatives = [hypothesis.code for hypothesis in hypotheses]
    if document_cause == leading:
        return 4
    if document_cause in RELATED_ROOT_CAUSES.get(leading, ()):
        return 3
    if document_cause in alternatives:
        return 2
    if any(document_cause in RELATED_ROOT_CAUSES.get(code, ()) for code in alternatives):
        return 1
    return 0


def _prioritize_knowledge(
    knowledge: KnowledgeRetrieval,
    hypotheses: list[Hypothesis],
) -> KnowledgeRetrieval:
    """Order past incidents by root-cause agreement first, then by retrieval score."""
    ranked_incidents = sorted(
        knowledge.similar_incidents,
        key=lambda item: (
            -_root_cause_relation(item.root_cause, hypotheses),
            -item.score,
            item.reference_id,
        ),
    )
    return knowledge.model_copy(update={"similar_incidents": tuple(ranked_incidents)})


def _knowledge_summary(knowledge: KnowledgeRetrieval, hypotheses: list[Hypothesis]) -> str:
    matches = [
        (item, _root_cause_relation(item.root_cause, hypotheses))
        for item in knowledge.similar_incidents
    ]
    matches = [(item, relation) for item, relation in matches if relation]
    if not matches:
        return ""
    described = ", ".join(
        f"{item.reference_id} ({item.root_cause}, "
        f"{'same cause' if relation in (2, 4) else 'related cause'})"
        for item, relation in matches
    )
    return f" Similar past incidents with a matching or related root cause: {described}."


def _qualifying_deployment_id(payload: DiagnosisInput) -> str | None:
    if payload.memory_rise_started_at is None:
        return None
    qualifying = [
        event
        for event in payload.recent_change_events
        if qualifies_as_prior_deployment(payload, event)
    ]
    if not qualifying:
        return None
    return max(qualifying, key=lambda event: event.changed_at).change_id


def _latest_dated_change_id(payload: DiagnosisInput) -> str | None:
    if not payload.recent_change_events:
        return None
    return max(payload.recent_change_events, key=lambda event: event.changed_at).change_id


# Weight of each provenance when judging how strong the supporting evidence is.
_PROVENANCE_WEIGHT = {"provided": 1.0, "derived": 0.7, "fixture": 0.5}


def evidence_strength(hypothesis: Hypothesis, evidence: list[EvidenceItem]) -> float:
    """Score 0-1 from the variety and provenance of the evidence citing a hypothesis.

    Three distinct evidence types (e.g. metric + log + k8s_event) give full variety; observed
    evidence counts more than evidence derived from input signals.
    """
    by_id = {item.evidence_id: item for item in evidence}
    supporting = [by_id[ref] for ref in hypothesis.evidence_refs if ref in by_id]
    if not supporting:
        return 0.0
    variety = min(len({item.evidence_type for item in supporting}) / 3, 1.0)
    provenance = sum(_PROVENANCE_WEIGHT[item.provenance] for item in supporting) / len(supporting)
    return round(0.6 * variety + 0.4 * provenance, 2)


def _validate_analysis(
    analysis: SupplementalAnalysis,
    hypotheses: list[Hypothesis],
    evidence: list[EvidenceItem],
) -> None:
    candidate_codes = {hypothesis.code for hypothesis in hypotheses}
    valid_evidence_ids = {item.evidence_id for item in evidence}
    if analysis.selected_cause is not None and analysis.selected_cause not in candidate_codes:
        raise ValueError("supplemental analyzer selected a cause absent from the rule candidates")
    if not set(analysis.evidence_refs) <= valid_evidence_ids:
        raise ValueError("supplemental analyzer referenced evidence absent from this diagnosis")


def _metric_is_available(payload: DiagnosisInput, *names: str) -> bool:
    signal_values = [payload.signals[name] for name in names if name in payload.signals]
    evidence_values = [
        item.value[name]
        for item in payload.evidence
        for name in names
        if name in item.value
    ]
    return any(
        isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
        for value in (*signal_values, *evidence_values)
    )


def _consistent_measure(
    payload: DiagnosisInput,
    name: str,
    conflicts: tuple[str, ...],
) -> float | None:
    if name in conflicts:
        return None
    for source in (payload.signals, *(item.value for item in payload.evidence)):
        value = source.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
        ):
            return float(value)
    return None


def _opposing_evidence_refs(
    hypothesis: Hypothesis,
    evidence: list[EvidenceItem],
) -> tuple[str, ...]:
    opposing: list[str] = []
    for item in evidence:
        if hypothesis.code == "MEMORY_LEAK_POST_DEPLOY":
            value = item.value.get("memory_percent", item.value.get("memory_pct"))
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 70:
                opposing.append(item.evidence_id)
    return tuple(dict.fromkeys(opposing))


def _hypothesis_missing_information(
    code: str,
    payload: DiagnosisInput,
) -> tuple[str, ...]:
    missing: list[str] = []
    if code in ("MEMORY_PRESSURE", "MEMORY_LEAK_POST_DEPLOY"):
        if not payload.memory_growth_observed and not any(
            "heap_profile" in item.value or "memory_trend" in item.value
            for item in payload.evidence
        ):
            missing.append(
                "A memory trend or heap profile is needed to distinguish a leak from "
                "temporary high utilization."
            )
        if payload.memory_rise_started_at is None:
            missing.append(
                "The beginning of the memory rise is not timestamped, so a change cannot be "
                "ordered before it."
            )
        elif not any(
            event.changed_at < payload.memory_rise_started_at
            for event in payload.recent_change_events
        ):
            missing.append(
                "No timestamped recent change is strictly before the beginning of the memory rise."
            )
        signal_restart_count = payload.signals.get("restart_count")
        has_restart = (
            isinstance(signal_restart_count, (int, float))
            and not isinstance(signal_restart_count, bool)
            and signal_restart_count > 0
        ) or any(
            isinstance(item.value.get("restart_count"), (int, float))
            and not isinstance(item.value.get("restart_count"), bool)
            and item.value["restart_count"] > 0
            for item in payload.evidence
        )
        has_oom = any(
            item.evidence_type in ("log", "k8s_event") and has_oom_observation(item)
            for item in payload.evidence
        )
        if not payload.memory_growth_observed and not has_restart and not has_oom:
            missing.append("No memory-growth, restart, or OOM indicator was supplied.")
        if code == "MEMORY_LEAK_POST_DEPLOY":
            missing.append(
                "Temporal ordering supports a post-change hypothesis but does not prove the "
                "change caused the memory rise."
            )
        if not any(
            item.evidence_type in ("log", "k8s_event")
            and any(
                token in f"{item.summary} {item.value.get('reason', '')}".lower()
                for token in ("oom", "oomkilled", "out of memory")
            )
            for item in payload.evidence
        ):
            missing.append("No OOM-specific log or Kubernetes event was supplied.")
    elif code == "MEMORY_GROWTH":
        missing.append(
            "A timestamped deployment history is required to assess whether the memory growth "
            "followed a deployment."
        )
        if _uncertain_deployment_correlation(payload):
            missing.append(
                "Deployment-to-detection timing is uncertain: the deployment preceded first_seen "
                "by less than the 5-minute detection margin."
            )
    elif code == "MEMORY_LIMIT_REACHED":
        missing.append(
            "The memory peak ratio reached the configured OOM threshold; confirm the effective "
            "memory limit and inspect OOM events before remediation."
        )
    elif code == "RESOURCE_PRESSURE_OR_CPU_STARVATION":
        if not _metric_is_available(payload, "cpu_percent"):
            missing.append("A CPU utilization measurement is absent.")
        if not _metric_is_available(payload, "error_rate", "error_rate_percent"):
            missing.append("An error-rate measurement is absent.")
        if not _metric_is_available(payload, "latency_ms"):
            missing.append("A latency measurement is absent.")
    else:
        for names, label in (
            (("memory_percent", "memory_pct"), "memory utilization"),
            (("cpu_percent",), "CPU utilization"),
            (("error_rate", "error_rate_percent"), "error rate"),
            (("latency_ms",), "latency"),
            (("restart_count",), "restart count"),
        ):
            if not _metric_is_available(payload, *names):
                missing.append(f"No {label} measurement was supplied.")
        if not missing:
            missing.append(
                "The supplied measurements do not meet any current deterministic rule "
                "combination."
            )
    return tuple(missing)


def _build_explainability(
    payload: DiagnosisInput,
    hypotheses: list[Hypothesis],
    evidence: list[EvidenceItem],
    recommendations: list[DiagnosisRecommendation],
) -> DiagnosisExplainability:
    conflicts = signal_conflicts(payload.model_copy(update={"evidence": evidence}))
    available_refs = {item.evidence_id for item in evidence}
    explanations: list[HypothesisExplanation] = []

    for hypothesis in hypotheses:
        supporting_refs = tuple(
            reference for reference in hypothesis.evidence_refs if reference in available_refs
        )
        missing = _hypothesis_missing_information(hypothesis.code, payload)
        if hypothesis.code == "INSUFFICIENT_EVIDENCE":
            explanation = (
                "Aucune combinaison de seuils des règles déterministes n’étaye actuellement "
                "une cause spécifique ; ce code décrit l’insuffisance des données, pas une cause."
            )
        elif hypothesis.code == "MEMORY_PRESSURE":
            explanation = (
                "Mémoire au-dessus du seuil de pression; aucune chronologie complète ne permet "
                "de rattacher cette hausse à un changement antérieur."
            )
        elif hypothesis.code == "MEMORY_LEAK_POST_DEPLOY":
            explanation = (
                "Hypothèse de hausse mémoire après changement : la mémoire est élevée, un signal "
                "de croissance/redémarrage/OOM est présent et un changement horodaté précède le "
                "début horodaté de la hausse. Cette chronologie ne prouve pas la causalité."
            )
        elif hypothesis.code == "MEMORY_GROWTH":
            explanation = (
                "Croissance mémoire observée avec une pente Agent 1 atteignant le seuil; "
                "aucune conclusion de fuite post-déploiement n'est tirée sans chronologie "
                "de déploiement suffisamment fiable."
            )
        elif hypothesis.code == "MEMORY_LIMIT_REACHED":
            explanation = (
                "Le ratio de pic mémoire de l'Agent 1 atteint le seuil d'incident OOM; "
                "le redémarrage est cité séparément lorsqu'il est disponible."
            )
        else:
            explanation = (
                "Candidat de pression de ressources, soutenu par les mesures CPU, latence ou "
                "taux d’erreur citées ; les règles ne démontrent pas à elles seules le mécanisme."
            )

        if len(hypotheses) == 1:
            ranking_reason = (
                f"Rang {hypothesis.rank} : seul candidat retourné par les règles actuelles; "
                f"score heuristique {hypothesis.probability:.2f}, non calibré."
            )
        else:
            higher_ranked = [
                item for item in hypotheses if item.rank < hypothesis.rank
            ]
            lower_ranked = [
                item for item in hypotheses if item.rank > hypothesis.rank
            ]
            comparison = []
            if higher_ranked:
                comparison.append(
                    "classé après "
                    + ", ".join(
                        f"{item.code} ({item.probability:.2f})" for item in higher_ranked
                    )
                )
            if lower_ranked:
                comparison.append(
                    "classé avant "
                    + ", ".join(
                        f"{item.code} ({item.probability:.2f})" for item in lower_ranked
                    )
                )
            ranking_reason = (
                f"Rang {hypothesis.rank} sur {len(hypotheses)} selon le score heuristique "
                f"{hypothesis.probability:.2f} (non calibré), trié par score décroissant puis "
                f"code en cas d’égalité; {'; '.join(comparison)}."
            )
        explanations.append(
            HypothesisExplanation(
                rank=hypothesis.rank,
                code=hypothesis.code,
                explanation=explanation,
                supporting_evidence_refs=supporting_refs,
                opposing_evidence_refs=_opposing_evidence_refs(hypothesis, evidence),
                missing_information=missing,
                ranking_reason=ranking_reason,
            )
        )

    recommendation_explanations = []
    primary = hypotheses[0]
    for recommendation in recommendations:
        if (
            recommendation.action == "ROLLBACK_OR_REVERT_DEPLOYMENT"
            and primary.code == "CONFIG_ERROR_CRASHLOOP"
        ):
            justification = (
                "Le conteneur ne démarre plus depuis un changement daté et les preuves pointent "
                "une configuration invalide; revenir à la version précédente restaure la "
                "configuration fonctionnelle."
            )
            caveats = (
                "Confirmer le changement concerné; l’exécution reste soumise à la politique de "
                "risque et à l’approbation humaine.",
            )
        elif recommendation.action == "ROLLBACK_OR_REVERT_DEPLOYMENT":
            justification = (
                "Un déploiement daté précède la montée mémoire d’au moins la marge de détection; "
                "revenir à la version précédente est la remédiation la plus directe. La "
                "corrélation temporelle ne prouve pas à elle seule la causalité."
            )
            caveats = (
                "Ne pas exécuter sans confirmer la régression, le changement concerné et une "
                "version cible connue comme saine.",
                "L’exécution reste soumise à la politique de risque et à l’approbation humaine.",
            )
        elif recommendation.action == "SCALE_OR_REBALANCE_RESOURCES":
            justification = (
                "Suggestion liée au candidat de pression de ressources; les données ne donnent "
                "pas le nombre de réplicas ni une capacité cible."
            )
            if primary.code == "MEMORY_LEAK_POST_DEPLOY":
                justification = (
                    "Alternative temporaire au rollback : ajouter de la capacité retarde les "
                    "redémarrages OOM mais ne corrige pas la fuite mémoire."
                )
            caveats = ("Action indicative seulement; les paramètres d’exécution restent incomplets.",)
        elif recommendation.action == "RESTART_TO_CONTAIN_MEMORY_GROWTH":
            justification = (
                "Un redémarrage progressif libère la mémoire accumulée et contient la croissance "
                "observée par l’agent 1 en attendant l’historique de déploiement."
            )
            caveats = (
                "Mesure de confinement temporaire : la croissance reprendra si la cause persiste.",
            )
        elif recommendation.action == "INCREASE_MEMORY_LIMIT":
            justification = (
                "Le pic mémoire a atteint la limite du conteneur (memory_peak_ratio ≥ 0.95) et "
                "un redémarrage a été observé; relever la limite évite les terminaisons OOM."
            )
            caveats = (
                "Aucun ActionType partagé ne couvre encore la modification de limite mémoire; "
                "l’action reste locale à l’agent 2.",
                "Vérifier d’abord qu’il ne s’agit pas d’une fuite (profil heap, historique de "
                "déploiement).",
            )
        elif recommendation.action == "FIX_CONFIGURATION":
            justification = (
                "Le conteneur redémarre en boucle et les logs signalent une configuration "
                "invalide; corriger la configuration est la seule remédiation durable."
            )
            caveats = (
                "Aucun ActionType partagé ne couvre la correction de configuration; l’action "
                "reste locale à l’agent 2.",
            )
        elif recommendation.action == "INVESTIGATE_DEPENDENCY":
            justification = (
                "Les erreurs de connexion ou les timeouts désignent une dépendance externe; le "
                "service diagnostiqué est la victime, pas la cause."
            )
            caveats = (
                "Ni rollback ni scaling de ce service ne sont recommandés; l’action reste locale "
                "à l’agent 2.",
            )
        elif recommendation.action == "TUNE_CONNECTION_POOL":
            justification = (
                "Les logs indiquent un pool de connexions épuisé; ajuster la taille du pool ou "
                "libérer les connexions bloquées réduit l’attente des requêtes."
            )
            caveats = (
                "Vérifier la capacité de la base avant d’augmenter le pool; l’action reste "
                "locale à l’agent 2.",
            )
        else:
            justification = (
                "Collecter les mesures et événements manquants avant de conclure ou de proposer "
                "une action de remédiation."
            )
            caveats = ("Cette recommandation ne constitue pas une action d’infrastructure.",)
        recommendation_explanations.append(
            RecommendationExplanation(
                action=recommendation.action,
                justification=justification,
                evidence_refs=primary.evidence_refs,
                limitations=caveats,
            )
        )

    all_missing = tuple(
        dict.fromkeys(
            information
            for explanation in explanations
            for information in explanation.missing_information
        )
    )
    return DiagnosisExplainability(
        observed_evidence_refs=tuple(
            item.evidence_id for item in evidence if item.provenance == "provided"
        ),
        derived_evidence_refs=tuple(
            item.evidence_id for item in evidence if item.provenance == "derived"
        ),
        fixture_context_refs=tuple(
            item.evidence_id for item in evidence if item.provenance == "fixture"
        ),
        missing_information=all_missing,
        contradictions=conflicts,
        hypothesis_explanations=tuple(explanations),
        recommendation_explanations=tuple(recommendation_explanations),
    )


def _with_collected(
    payload: DiagnosisInput,
    collected: tuple[EvidenceItem, ...],
) -> DiagnosisInput:
    """Add tool-collected evidence and the dated changes it describes to the input."""
    known_evidence = {item.evidence_id for item in payload.evidence}
    known_changes = {event.change_id for event in payload.recent_change_events}
    new_evidence = [item for item in collected if item.evidence_id not in known_evidence]
    new_changes = [
        event
        for event in change_events_from_evidence(tuple(new_evidence))
        if event.change_id not in known_changes
    ]
    return payload.model_copy(
        update={
            "evidence": [*payload.evidence, *new_evidence],
            "recent_change_events": [*payload.recent_change_events, *new_changes],
        }
    )


def _apply_supplemental_ranking(
    analysis: SupplementalAnalysis,
    hypotheses: list[Hypothesis],
    limitations: list[str],
) -> tuple[list[Hypothesis], bool]:
    """Let a language model reorder close rule candidates; report larger disagreements."""
    selected = analysis.selected_cause
    top = hypotheses[0]
    if analysis.source != "llm" or selected is None or selected == top.code:
        return hypotheses, False
    chosen = next(hypothesis for hypothesis in hypotheses if hypothesis.code == selected)
    if top.probability - chosen.probability <= LLM_RERANK_MARGIN:
        reordered = [chosen, *(item for item in hypotheses if item.code != selected)]
        limitations.append(
            f"The language-model analysis preferred {selected} over {top.code}; their rule "
            f"scores differ by at most {LLM_RERANK_MARGIN}, so the ranking was adjusted."
        )
        return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(reordered, 1)], False
    limitations.append(
        f"The language-model analysis preferred {selected}, but the rules rank {top.code} "
        "clearly higher; the rule ranking was kept and the disagreement needs review."
    )
    return hypotheses, True


def build_result(
    payload: DiagnosisInput,
    retriever: KnowledgeRetriever | None = None,
    analyzer: SupplementalAnalyzer | None = None,
    toolbox: Toolbox | None = None,
    deadline: datetime | None = None,
) -> DiagnosisResult:
    """Collect targeted evidence, apply deterministic rules, then the isolated adapters."""

    limitations = [
        "Hypothesis scores are heuristic ranking scores and are not calibrated probabilities.",
        "Recommendation scores are heuristic and are not calibrated confidence estimates.",
    ]
    deadline = deadline or datetime.now(UTC) + DEFAULT_INVESTIGATION_TIMEOUT
    tool_calls: list[ToolCallRecord] = []
    if toolbox is not None:
        investigation = investigate(payload, toolbox, deadline)
        payload = _with_collected(payload, investigation.evidence)
        tool_calls.extend(investigation.tool_calls)
        limitations.extend(investigation.limitations)
    elif payload.required_evidence:
        limitations.append(
            "Requested evidence could not be collected because no evidence tools are configured: "
            + ", ".join(payload.required_evidence)
            + "."
        )

    evidence = collect_evidence(payload)
    evidence_payload = payload.model_copy(update={"evidence": evidence})
    hypotheses = evaluate_hypotheses(evidence_payload)

    if not evidence:
        limitations.append("No evidence was available from the supplied diagnosis input.")

    knowledge = KnowledgeRetrieval()
    try:
        active_retriever = retriever or LocalKnowledgeRetriever()
        knowledge = _prioritize_knowledge(
            KnowledgeRetrieval.model_validate(
                active_retriever.retrieve(
                    _retrieval_query(payload, evidence, hypotheses), RETRIEVAL_CANDIDATES
                )
            ),
            hypotheses,
        )
        # Fetch more than needed so root-cause agreement can promote a lower-scored match.
        knowledge = knowledge.model_copy(
            update={
                "similar_incidents": knowledge.similar_incidents[:KNOWLEDGE_LIMIT],
                "runbooks": knowledge.runbooks[:KNOWLEDGE_LIMIT],
            }
        )
    except Exception as exc:
        limitations.append(
            f"Knowledge retrieval failed ({type(exc).__name__}); deterministic rules continued."
        )

    active_analyzer = analyzer or DeterministicSupplementalAnalyzer()
    disagreement = False
    try:
        analysis = SupplementalAnalysis.model_validate(
            active_analyzer.analyze(
                payload,
                tuple(hypotheses),
                tuple(evidence),
                knowledge,
            )
        )
        tool_calls.extend(analysis.tool_calls)
        if analysis.collected_evidence:
            # Evidence found by the analyzer goes back through the deterministic rules.
            payload = _with_collected(payload, analysis.collected_evidence)
            evidence = collect_evidence(payload)
            evidence_payload = payload.model_copy(update={"evidence": evidence})
            hypotheses = evaluate_hypotheses(evidence_payload)
        _validate_analysis(analysis, hypotheses, evidence)
        hypotheses, disagreement = _apply_supplemental_ranking(analysis, hypotheses, limitations)
        if analysis.selected_cause is not None:
            limitations.extend(analysis.limitations)
    except Exception as exc:
        limitations.append(
            f"Supplemental analysis failed validation or execution ({type(exc).__name__}: "
            f"{str(exc)[:200]}); "
            "deterministic rules continued."
        )
        analysis = None

    primary_root_cause = hypotheses[0].code
    limitations.extend(signal_conflicts(evidence_payload))

    recommendations: list[DiagnosisRecommendation] = []
    if primary_root_cause == "MEMORY_LEAK_POST_DEPLOY":
        deployment_id = _qualifying_deployment_id(evidence_payload)
        rollback_parameters: dict[str, object] = {"service": payload.service_id}
        if deployment_id is not None:
            rollback_parameters["deployment_id"] = deployment_id
        recommendations.extend(
            [
                DiagnosisRecommendation(
                    action="ROLLBACK_OR_REVERT_DEPLOYMENT",
                    parameters=rollback_parameters,
                    score=0.85,
                ),
                DiagnosisRecommendation(
                    action="COLLECT_MORE_EVIDENCE",
                    parameters={"missing_signals": ["heap_profile", "previous_stable_version"]},
                    score=0.6,
                ),
                DiagnosisRecommendation(
                    action="SCALE_OR_REBALANCE_RESOURCES",
                    parameters={
                        "service": payload.service_id,
                        "expected_effect": "temporary_capacity_only",
                    },
                    score=0.4,
                ),
            ]
        )
        limitations.append(
            "The dated deployment precedes the memory rise by at least the detection margin; "
            "this supports but does not prove causality. Rollback is a recommendation only and "
            "its target version must be confirmed before approval."
        )
    elif primary_root_cause == "MEMORY_GROWTH":
        recommendations.extend(
            [
                DiagnosisRecommendation(
                    action="COLLECT_MORE_EVIDENCE",
                    parameters={
                        "missing_signals": [
                            "deployment_timeline",
                            "heap_profile",
                            "memory_limit",
                        ]
                    },
                    score=0.65,
                ),
                DiagnosisRecommendation(
                    action="RESTART_TO_CONTAIN_MEMORY_GROWTH",
                    parameters={
                        "service": payload.service_id,
                        "expected_effect": "temporary_containment_only",
                    },
                    score=0.5,
                ),
            ]
        )
        limitations.append(
            "Memory growth is supported by Agent 1 facts; deployment history is absent or "
            "within the 5-minute detection uncertainty margin, so no post-deployment leak "
            "or rollback is inferred."
        )
    elif primary_root_cause == "MEMORY_PRESSURE":
        recommendations.append(
            DiagnosisRecommendation(
                action="COLLECT_MORE_EVIDENCE",
                parameters={
                    "missing_signals": [
                        "memory_limit",
                        "memory_trend",
                        "memory_rise_started_at",
                        "recent_change_events",
                        "deployment_timeline",
                        "heap_profile",
                        "oom_events",
                    ]
                },
                score=0.65,
            )
        )
        limitations.append(
            "Memory pressure is observed without the complete dated evidence required for a "
            "post-deployment hypothesis."
        )
    elif primary_root_cause == "MEMORY_LIMIT_REACHED":
        recommendations.extend(
            [
                DiagnosisRecommendation(
                    action="INCREASE_MEMORY_LIMIT",
                    parameters={"service": payload.service_id},
                    score=0.7,
                ),
                DiagnosisRecommendation(
                    action="COLLECT_MORE_EVIDENCE",
                    parameters={
                        "missing_signals": [
                            "memory_limit",
                            "deployment_timeline",
                            "heap_profile",
                        ]
                    },
                    score=0.65,
                ),
            ]
        )
        limitations.append(
            "Agent 1 memory_oom ratio reached the configured threshold; no rollback is "
            "recommended without a dated deployment event."
        )
    elif primary_root_cause == "CONFIG_ERROR_CRASHLOOP":
        latest_change = _latest_dated_change_id(evidence_payload)
        if latest_change is not None:
            recommendations.append(
                DiagnosisRecommendation(
                    action="ROLLBACK_OR_REVERT_DEPLOYMENT",
                    parameters={
                        "service": payload.service_id,
                        "deployment_id": latest_change,
                        "expected_effect": "restore_previous_working_configuration",
                    },
                    score=0.8,
                )
            )
        else:
            recommendations.append(
                DiagnosisRecommendation(
                    action="FIX_CONFIGURATION",
                    parameters={
                        "service": payload.service_id,
                        "expected_effect": "container_starts_with_valid_configuration",
                    },
                    score=0.7,
                )
            )
        recommendations.append(
            DiagnosisRecommendation(
                action="COLLECT_MORE_EVIDENCE",
                parameters={"missing_signals": ["config_diff", "previous_container_logs"]},
                score=0.6,
            )
        )
        limitations.append(
            "Restarting or scaling a container that fails on its configuration does not fix it; "
            "the configuration change must be reverted or corrected."
        )
    elif primary_root_cause == "CRASH_LOOP_UNEXPLAINED":
        recommendations.append(
            DiagnosisRecommendation(
                action="COLLECT_MORE_EVIDENCE",
                parameters={
                    "missing_signals": ["previous_container_logs", "config_diff", "exit_code"]
                },
                score=0.65,
            )
        )
    elif primary_root_cause in ("DEPENDENCY_UNAVAILABLE", "CONNECTION_POOL_EXHAUSTION"):
        dependency = _suspected_dependency(evidence_payload)
        if primary_root_cause == "CONNECTION_POOL_EXHAUSTION":
            recommendations.append(
                DiagnosisRecommendation(
                    action="TUNE_CONNECTION_POOL",
                    parameters={
                        "service": payload.service_id,
                        **({"dependency": dependency} if dependency else {}),
                        "expected_effect": "requests_obtain_connections_without_waiting",
                    },
                    score=0.7,
                )
            )
        else:
            recommendations.append(
                DiagnosisRecommendation(
                    action="INVESTIGATE_DEPENDENCY",
                    parameters={
                        "service": payload.service_id,
                        **({"dependency": dependency} if dependency else {}),
                        "expected_effect": "restore_dependency_availability",
                    },
                    score=0.75,
                )
            )
        recommendations.append(
            DiagnosisRecommendation(
                action="COLLECT_MORE_EVIDENCE",
                parameters={
                    "missing_signals": ["dependency_health", "connection_metrics"],
                    **({"dependency": dependency} if dependency else {}),
                },
                score=0.6,
            )
        )
        limitations.append(
            "The failure originates outside this service; rolling back or scaling it is not "
            "expected to help and is not recommended."
        )
    elif primary_root_cause == "TRAFFIC_SPIKE":
        recommendations.append(
            DiagnosisRecommendation(
                action="SCALE_OR_REBALANCE_RESOURCES",
                parameters={
                    "service": payload.service_id,
                    "expected_effect": "absorb_increased_traffic",
                },
                score=0.75,
            )
        )
        limitations.append(
            "The scaling recommendation does not specify a replica count; it is not an execution plan."
        )
    elif primary_root_cause == "RESOURCE_PRESSURE_OR_CPU_STARVATION":
        conflicts = conflicting_metrics(evidence_payload)
        cpu = _consistent_measure(evidence_payload, "cpu_percent", conflicts)
        latency = _consistent_measure(evidence_payload, "latency_ms", conflicts)
        if cpu is not None and (cpu >= 70 or (cpu >= 50 and latency is not None and latency >= 800)):
            recommendations.append(
                DiagnosisRecommendation(
                    action="SCALE_OR_REBALANCE_RESOURCES",
                    parameters={"service": payload.service_id},
                    score=0.76,
                )
            )
            limitations.append(
                "The scaling recommendation does not specify a replica count; it is not an execution plan."
            )
        else:
            recommendations.append(
                DiagnosisRecommendation(
                    action="COLLECT_MORE_EVIDENCE",
                    parameters={"missing_signals": ["cpu_percent", "latency_ms"]},
                    score=0.65,
                )
            )
            limitations.append(
                "A high error rate alone does not justify scaling without CPU or correlated latency evidence."
            )
    else:
        recommendations.append(
            DiagnosisRecommendation(
                action="COLLECT_MORE_EVIDENCE",
                parameters={"missing_signals": ["heap_profile", "recent_logs", "config_diff"]},
                score=0.65,
            )
        )

    cited_evidence = tuple(
        dict.fromkeys(
            evidence_ref
            for hypothesis in hypotheses
            for evidence_ref in hypothesis.evidence_refs
        )
    )
    summary = f"Deterministic rules ranked {primary_root_cause} as the leading candidate."
    if cited_evidence:
        summary += f" Supporting input evidence references: {', '.join(cited_evidence)}."
    else:
        summary += " No current-incident evidence reference supports a specific cause."
    knowledge_refs = tuple(
        item.reference_id for item in (*knowledge.similar_incidents, *knowledge.runbooks)
    )
    if knowledge_refs:
        summary += f" Retrieved knowledge context references: {', '.join(knowledge_refs)}."
        summary += _knowledge_summary(knowledge, hypotheses)
    if analysis is not None and analysis.selected_cause == primary_root_cause:
        summary += " The local supplemental analyzer agrees with the leading rule candidate."

    explainability = _build_explainability(payload, hypotheses, evidence, recommendations)
    if explainability.contradictions:
        summary += " Contradictory threshold readings were detected: " + "; ".join(
            explainability.contradictions
        )
    for explanation in explainability.hypothesis_explanations:
        summary += f" {explanation.ranking_reason} {explanation.explanation}"

    top_score = hypotheses[0].probability
    # Missing nice-to-have data stays in explainability; only gaps that prevent acting count.
    actionable = any(item.action != "COLLECT_MORE_EVIDENCE" for item in recommendations)
    additional_evidence = (
        primary_root_cause in SYMPTOM_LEVEL_CAUSES
        or top_score < 0.55
        or not hypotheses[0].evidence_refs
        or bool(signal_conflicts(evidence_payload))
        or disagreement
        or not actionable
    )
    if additional_evidence:
        limitations.append("Additional evidence is needed before treating this result as conclusive.")

    return DiagnosisResult(
        diagnosis_id=f"DIAG-{payload.incident_id}",
        incident_id=payload.incident_id,
        primary_root_cause=primary_root_cause,
        hypotheses=hypotheses,
        recommended_actions=recommendations,
        rationale_summary=summary,
        limitations=list(dict.fromkeys(limitations)),
        requires_additional_evidence=additional_evidence,
        evidence=tuple(evidence),
        knowledge_refs=knowledge_refs,
        knowledge=knowledge,
        explainability=explainability,
        evidence_strength=evidence_strength(hypotheses[0], evidence),
        tool_calls=tuple(tool_calls),
        supplemental_analysis=analysis,
    )


def to_diagnosis_completed(task: AgentTask, result: DiagnosisResult) -> DiagnosisCompleted:
    """Convert the diagnosis result to shared contracts without trusting payload identifiers."""

    task_context = task.task_context
    actions: list[ActionIntent] = []
    limitations = list(result.limitations)
    requires_additional_evidence = result.requires_additional_evidence

    for recommendation in result.recommended_actions:
        action_type: ActionType | None
        if recommendation.action == "ROLLBACK_OR_REVERT_DEPLOYMENT":
            action_type = ActionType.ROLLBACK
        elif recommendation.action == "SCALE_OR_REBALANCE_RESOURCES":
            action_type = ActionType.SCALE_REPLICAS
        elif recommendation.action == "RESTART_TO_CONTAIN_MEMORY_GROWTH":
            action_type = ActionType.ROLLING_RESTART
        elif recommendation.action == "COLLECT_MORE_EVIDENCE":
            action_type = None
            if requires_additional_evidence:
                limitations.append("Additional evidence is required before recommending an action.")
        else:
            action_type = None
            limitations.append(
                f"Recommended action '{recommendation.action}' has no equivalent in the shared action contract."
            )

        if action_type is not None:
            action_number = len(actions) + 1
            actions.append(
                ActionIntent(
                    action_id=f"ACTION-{task.task_id}-{action_number}",
                    incident_id=task.incident_id,
                    action=action_type,
                    environment=task_context.environment,
                    target=ActionTarget(
                        service=task_context.service_id,
                        namespace=task_context.namespace,
                    ),
                    parameters=recommendation.parameters,
                    verification_profile=f"{task_context.service_id}-slo-default",
                )
            )

    if actions:
        limitations.append(
            "The verification profile is a local placeholder and requires team confirmation."
        )

    return DiagnosisCompleted(
        diagnosis_id=f"DIAG-{task.incident_id}",
        hypotheses=tuple(
            Hypothesis(
                rank=hypothesis.rank,
                code=hypothesis.code,
                probability=hypothesis.probability,
                evidence_refs=hypothesis.evidence_refs,
            )
            for hypothesis in result.hypotheses
        ),
        primary_root_cause=result.primary_root_cause,
        recommended_actions=tuple(actions),
        rationale_summary=result.rationale_summary,
        limitations=tuple(limitations),
        requires_additional_evidence=requires_additional_evidence,
    )
