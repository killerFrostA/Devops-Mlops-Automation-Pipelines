import pytest
from pydantic import ValidationError

from src.contracts.payloads import DiagnosisCompleted, Hypothesis


def test_hypothesis_accepts_empty_evidence_refs() -> None:
    # Le contrat officiel autorise une liste vide (ex. INSUFFICIENT_EVIDENCE)
    h = Hypothesis(rank=1, code="INSUFFICIENT_EVIDENCE", probability=0.2, evidence_refs=())
    assert h.evidence_refs == ()


def test_hypothesis_requires_evidence_refs_field() -> None:
    # Le champ doit exister, même s'il peut être vide
    with pytest.raises(ValidationError, match="evidence_refs"):
        Hypothesis(rank=1, code="MEMORY_LEAK_POST_DEPLOY", probability=0.86)


def test_diagnosis_completed_requires_primary_root_cause_and_hypotheses() -> None:
    diagnosis = DiagnosisCompleted(
        diagnosis_id="DIAG-001",
        primary_root_cause="MEMORY_LEAK_POST_DEPLOY",
        hypotheses=(
            Hypothesis(
                rank=1,
                code="MEMORY_LEAK_POST_DEPLOY",
                probability=0.86,
                evidence_refs=("EVD-METRIC-01", "EVD-GIT-01"),
            ),
        ),
        recommended_actions=(),
        rationale_summary="Memory rose after deploy and restarts occurred.",
        limitations=("Heap profile unavailable",),
        requires_additional_evidence=True,
    )

    assert diagnosis.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    assert diagnosis.hypotheses[0].rank == 1