"""Run the diagnosis pipeline against four illustrative incidents."""

from __future__ import annotations

import sys

from src.agents.diagnosis.contracts import DiagnosisInput
from src.agents.diagnosis.evidence_collector import collect_evidence
from src.agents.diagnosis.result_builder import build_result


SIGNAL_LABELS = {
    "memory_percent": "mémoire (%)",
    "memory_pct": "mémoire (%)",
    "cpu_percent": "CPU (%)",
    "restart_count": "redémarrages",
    "latency_ms": "latence (ms)",
    "error_rate": "taux d’erreur",
}

INCIDENTS = (
    {
        "incident_id": "DEMO-001",
        "title": "Mémoire élevée après déploiement",
        "issue_summary": "La mémoire augmente et le service redémarre après un déploiement.",
        "signals": {"memory_percent": 94, "restart_count": 3},
        "recent_changes": ["Déploiement récent il y a 9 minutes (deploy v2.6)"],
    },
    {
        "incident_id": "DEMO-002",
        "title": "CPU élevé et latence importante",
        "issue_summary": "Le CPU est saturé et les requêtes répondent lentement.",
        "signals": {"cpu_percent": 95, "latency_ms": 850},
        "recent_changes": [],
    },
    {
        "incident_id": "DEMO-003",
        "title": "Service dans les valeurs normales",
        "issue_summary": "Aucune anomalie signalée.",
        "signals": {"memory_percent": 40, "cpu_percent": 30, "restart_count": 0},
        "recent_changes": [],
    },
    {
        "incident_id": "DEMO-004",
        "title": "Signal vague : latence légèrement élevée",
        "issue_summary": "La latence semble un peu élevée, sans autre anomalie connue.",
        "signals": {"latency_ms": 420},
        "recent_changes": [],
    },
)

CAUSE_LABELS = {
    "MEMORY_LEAK_POST_DEPLOY": "Fuite mémoire après déploiement",
    "RESOURCE_PRESSURE_OR_CPU_STARVATION": "Pression sur les ressources ou manque de CPU",
    "INSUFFICIENT_EVIDENCE": "Éléments insuffisants pour conclure",
}

ACTION_LABELS = {
    "ROLLBACK_OR_REVERT_DEPLOYMENT": "Revenir à la version précédente",
    "INCREASE_MEMORY_LIMIT_AND_VALIDATE_HEAP_PROFILE": (
        "Augmenter la limite mémoire et analyser le profil du tas"
    ),
    "SCALE_OR_REBALANCE_RESOURCES": "Augmenter ou rééquilibrer les ressources",
    "COLLECT_MORE_EVIDENCE": "Collecter davantage de preuves",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for incident in INCIDENTS:
        payload = DiagnosisInput(
            incident_id=incident["incident_id"],
            service_id="demo-api",
            namespace="demo",
            environment="test",
            issue_summary=incident["issue_summary"],
            signals=incident["signals"],
            recent_changes=incident["recent_changes"],
        )
        result = build_result(payload)
        evidence_by_id = {
            evidence.evidence_id: evidence
            for evidence in collect_evidence(payload)
        }
        used_evidence_ids = {
            evidence_id
            for hypothesis in result.hypotheses
            for evidence_id in hypothesis.evidence_refs
        }

        print(f"\n{'=' * 72}")
        print(f"Incident {payload.incident_id} — {incident['title']}")
        print("Signaux fournis :")
        for signal, value in payload.signals.items():
            label = SIGNAL_LABELS.get(signal, signal.replace("_", " "))
            print(f"  - {label} : {value}")
        if payload.recent_changes:
            for change in payload.recent_changes:
                print(f"  - changement récent : {change}")
        else:
            print("  - aucun déploiement ou changement récent fourni")

        cause = CAUSE_LABELS.get(result.primary_root_cause, result.primary_root_cause)
        print(f"Cause principale trouvée : {cause} ({result.primary_root_cause})")
        print("Hypothèses classées :")
        for hypothesis in result.hypotheses:
            label = CAUSE_LABELS.get(hypothesis.code, hypothesis.code)
            print(f"  {hypothesis.rank}. {label} — score {hypothesis.probability:.0%}")

        print("Preuves utilisées / collectées :")
        if evidence_by_id:
            for evidence_id, evidence in evidence_by_id.items():
                status = (
                    "référencée par une hypothèse"
                    if evidence_id in used_evidence_ids
                    else "collectée, non référencée par les hypothèses"
                )
                print(f"  - [{evidence_id}] ({status}) {evidence.summary}")
        else:
            print("  - aucune preuve collectée")

        print("Action recommandée :")
        if result.recommended_actions:
            for recommendation in result.recommended_actions:
                label = ACTION_LABELS.get(recommendation.action, recommendation.action)
                print(f"  - {label} (score {recommendation.score:.0%})")
        else:
            print("  - aucune")
        print(
            "Preuves supplémentaires nécessaires : "
            f"{'oui' if result.requires_additional_evidence else 'non'}"
        )


if __name__ == "__main__":
    main()
