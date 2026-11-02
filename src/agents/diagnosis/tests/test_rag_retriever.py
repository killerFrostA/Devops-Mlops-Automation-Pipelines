import json
from pathlib import Path

import pytest

from src.agents.diagnosis.contracts import DiagnosisInput, KnowledgeDocument, KnowledgeRetrieval
from src.agents.diagnosis import rag_retriever
from src.agents.diagnosis.rag_retriever import (
    LocalKnowledgeRetriever,
    retrieve_runbooks,
    retrieve_similar_incidents,
)
from src.agents.diagnosis.result_builder import build_result


def test_valid_json_corpora_load_and_lexical_ranking_is_exact_and_deterministic() -> None:
    retrieval = LocalKnowledgeRetriever().retrieve("memory restart latency", limit=3)

    assert [(item.reference_id, item.score) for item in retrieval.similar_incidents] == [
        ("LOCAL-INC-120", 0.75),
        ("LOCAL-INC-122", 0.25),
        ("LOCAL-INC-123", 0.25),
    ]
    assert [(item.reference_id, item.score) for item in retrieval.runbooks] == [
        ("LOCAL-RUNBOOK-01", 0.5),
        ("LOCAL-RUNBOOK-03", 0.25),
    ]
    assert all(item.provenance == "local_fixture" for item in retrieval.similar_incidents)
    assert all(item.provenance == "local_fixture" for item in retrieval.runbooks)


def test_missing_corpus_file_raises_a_path_specific_error(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="incidents.json"):
        LocalKnowledgeRetriever(tmp_path)


def test_malformed_json_corpus_raises_instead_of_returning_empty_results(tmp_path: Path) -> None:
    (tmp_path / "incidents.json").write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON.*incidents.json"):
        LocalKnowledgeRetriever(tmp_path)


def test_invalid_corpus_shape_raises_a_validation_error(tmp_path: Path) -> None:
    (tmp_path / "incidents.json").write_text(
        json.dumps({"format_version": 2, "documents": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid incident knowledge corpus"):
        LocalKnowledgeRetriever(tmp_path)


def test_duplicate_reference_ids_across_corpora_are_rejected(tmp_path: Path) -> None:
    knowledge_dir = Path(rag_retriever.__file__).with_name("knowledge")
    incidents = json.loads((knowledge_dir / "incidents.json").read_text(encoding="utf-8"))
    runbooks = json.loads((knowledge_dir / "runbooks.json").read_text(encoding="utf-8"))
    runbooks["documents"][0]["reference_id"] = incidents["documents"][0]["reference_id"]
    (tmp_path / "incidents.json").write_text(json.dumps(incidents), encoding="utf-8")
    (tmp_path / "runbooks.json").write_text(json.dumps(runbooks), encoding="utf-8")

    with pytest.raises(ValueError, match="unique across"):
        LocalKnowledgeRetriever(tmp_path)


def test_diagnosis_reports_corpus_loading_errors_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rag_retriever, "_DEFAULT_KNOWLEDGE_DIR", tmp_path)
    payload = DiagnosisInput(
        incident_id="INC-MISSING-KNOWLEDGE",
        service_id="payments-api",
        namespace="test",
        environment="test",
        issue_summary="CPU is saturated.",
        signals={"cpu_percent": 90},
    )

    result = build_result(payload)

    assert result.primary_root_cause == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert result.knowledge_refs == ()
    assert any("Knowledge retrieval failed (OSError)" in item for item in result.limitations)


def test_limit_and_incident_runbook_separation_are_exact() -> None:
    retrieval = LocalKnowledgeRetriever().retrieve("cpu errors saturation", limit=1)

    assert [
        (item.reference_id, item.kind, item.score)
        for item in retrieval.similar_incidents
    ] == [("LOCAL-INC-121", "incident", 0.75)]
    assert [
        (item.reference_id, item.kind, item.score)
        for item in retrieval.runbooks
    ] == [("LOCAL-RUNBOOK-02", "runbook", 0.6)]


def test_empty_unmatched_and_nonpositive_limit_queries_return_no_documents() -> None:
    retriever = LocalKnowledgeRetriever()

    assert retriever.retrieve("  ", limit=3) == KnowledgeRetrieval()
    assert retriever.retrieve("unrelated planetary weather", limit=3) == KnowledgeRetrieval()
    assert retriever.retrieve("memory restart", limit=0) == KnowledgeRetrieval()
    assert retrieve_similar_incidents("", limit=3) == []
    assert retrieve_runbooks("unrelated planetary weather", limit=3) == []


class KnowledgeOnlyRetriever:
    def retrieve(self, query: str, limit: int) -> KnowledgeRetrieval:
        return KnowledgeRetrieval(
            similar_incidents=(
                KnowledgeDocument(
                    reference_id="RAG-ONLY-INCIDENT",
                    kind="incident",
                    summary="A synthetic historical memory incident.",
                    root_cause="MEMORY_LEAK_POST_DEPLOY",
                    score=1.0,
                ),
            ),
            runbooks=(
                KnowledgeDocument(
                    reference_id="RAG-ONLY-RUNBOOK",
                    kind="runbook",
                    summary="A synthetic memory diagnostic runbook.",
                    score=1.0,
                ),
            ),
        )


def test_retrieved_knowledge_never_becomes_current_incident_evidence() -> None:
    payload = DiagnosisInput(
        incident_id="INC-KNOWLEDGE-BOUNDARY",
        service_id="payments-api",
        namespace="test",
        environment="test",
        issue_summary="Memory usage is elevated and the service restarted.",
        signals={"memory_percent": 88, "restart_count": 1},
    )

    result = build_result(payload, retriever=KnowledgeOnlyRetriever())
    current_evidence_ids = {item.evidence_id for item in result.evidence}
    hypothesis_evidence_refs = {
        reference
        for hypothesis in result.hypotheses
        for reference in hypothesis.evidence_refs
    }
    retrieved_documents = KnowledgeOnlyRetriever().retrieve("memory", 3)
    knowledge_ids = {
        item.reference_id
        for item in (*retrieved_documents.similar_incidents, *retrieved_documents.runbooks)
    }

    assert result.knowledge_refs == ("RAG-ONLY-INCIDENT", "RAG-ONLY-RUNBOOK")
    assert knowledge_ids.isdisjoint(current_evidence_ids)
    assert knowledge_ids.isdisjoint(hypothesis_evidence_refs)
    assert hypothesis_evidence_refs <= current_evidence_ids
