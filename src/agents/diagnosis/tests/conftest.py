from datetime import UTC, datetime

import pytest

from src.agents.diagnosis.contracts import EvidenceItem


@pytest.fixture
def operational_evidence_fixture() -> list[EvidenceItem]:
    """Representative test fixtures; every source is explicitly marked as synthetic."""

    observed_at = datetime(2026, 10, 12, 10, 15, tzinfo=UTC)
    return [
        EvidenceItem(
            evidence_id="FIX-METRIC-01",
            evidence_type="metric",
            source="fixture://prometheus",
            provenance="fixture",
            observed_at=observed_at,
            summary="Fixture: memory utilization reached 88% over the sample window.",
            value={"memory_percent": 88.0},
            confidence=0.9,
        ),
        EvidenceItem(
            evidence_id="FIX-LOG-01",
            evidence_type="log",
            source="fixture://loki",
            provenance="fixture",
            observed_at=observed_at,
            summary="Fixture: worker log contains an out-of-memory termination message.",
            value={"message": "fixture OOMKilled sample"},
            confidence=0.8,
        ),
        EvidenceItem(
            evidence_id="FIX-K8S-01",
            evidence_type="k8s_event",
            source="fixture://kubernetes",
            provenance="fixture",
            observed_at=observed_at,
            summary="Fixture: a pod restart event followed the synthetic memory alert.",
            value={"reason": "BackOff", "restart_count": 3},
            confidence=0.85,
        ),
        EvidenceItem(
            evidence_id="FIX-GIT-01",
            evidence_type="git_change",
            source="fixture://git",
            provenance="fixture",
            observed_at=observed_at,
            summary="Fixture: deployment v2.6 was published shortly before the alert.",
            value={"revision": "fixture-v2.6"},
            confidence=0.75,
        ),
    ]
