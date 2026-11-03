import json
from pathlib import Path

import pytest

from src.agents.diagnosis.agent1_runner import build_agent, run_agent1_file
from src.agents.diagnosis.evaluation.evaluate import (
    load_scenarios,
    render_markdown,
    run_evaluation,
)
from src.agents.diagnosis.qdrant_retriever import (
    COLLECTIONS,
    HashingEmbedder,
    QdrantKnowledgeRetriever,
    index_knowledge,
)
from src.agents.diagnosis.rag_retriever import load_corpus
from src.agents.diagnosis.tools import ToolError


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def qdrant() -> QdrantKnowledgeRetriever:
    return QdrantKnowledgeRetriever.in_memory()


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimension=64)
    first, second = embedder.embed(["connection refused", "connection refused"])

    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert embedder.embed([""])[0] == [0.0] * 64


def test_qdrant_retrieves_similar_incidents_and_runbooks(qdrant) -> None:
    result = qdrant.retrieve("database connection refused timeout errors dependency unavailable", 3)

    assert result.similar_incidents[0].root_cause == "DEPENDENCY_UNAVAILABLE"
    assert result.runbooks and result.runbooks[0].reference_id == "LOCAL-RUNBOOK-12"
    assert all(item.provenance == "qdrant" for item in (*result.similar_incidents, *result.runbooks))
    assert qdrant.retrieve("   ", 3).similar_incidents == ()
    assert qdrant.retrieve("memory", 0).runbooks == ()


def test_on_disk_store_can_be_indexed_and_reopened(tmp_path: Path) -> None:
    store = tmp_path / "qdrant"
    retriever = QdrantKnowledgeRetriever.connect(path=store)
    counts = index_knowledge(retriever.client)
    retriever.client.close()

    reopened = QdrantKnowledgeRetriever.connect(path=store)
    incidents, runbooks = load_corpus()  # grows with lab captures (incidents_lab.json)
    assert counts == {COLLECTIONS["incident"]: len(incidents), COLLECTIONS["runbook"]: len(runbooks)}
    assert reopened.retrieve("container oom memory limit reached", 1).similar_incidents
    reopened.client.close()
    with pytest.raises(ValueError, match="exactly one"):
        QdrantKnowledgeRetriever.connect()


def test_agent_with_qdrant_reports_vector_knowledge(tmp_path: Path) -> None:
    agent = build_agent(rag="qdrant")
    [written] = run_agent1_file(FIXTURES / "agent1_memory_oom.json", tmp_path, agent=agent)

    knowledge = written.diagnosis.result.knowledge
    assert knowledge.similar_incidents[0].root_cause == "MEMORY_LIMIT_REACHED"
    assert knowledge.similar_incidents[0].provenance == "qdrant"


def test_runner_replays_recorded_tool_responses(tmp_path: Path) -> None:
    responses = tmp_path / "tools.json"
    responses.write_text(json.dumps({"tools": {"kubernetes": {"deployment_history": [{
        "evidence_type": "k8s_event",
        "summary": "Deployment revision 42 rolled out.",
        "value": {"change_id": "payment-service-rev42", "description": "Deployment revision 42",
                  "changed_at": "2026-10-20T11:09:00+00:00"},
    }]}}}), encoding="utf-8")

    [written] = run_agent1_file(
        FIXTURES / "agent1_memory_pressure.json",
        tmp_path / "out",
        agent=build_agent(tool_responses=responses),
    )

    assert written.diagnosis.result.diagnosis.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    with pytest.raises(Exception, match="--live-tools needs"):
        build_agent(live_tools=True)
    with pytest.raises(ToolError):
        build_agent(tool_responses=tmp_path / "missing.json")


def test_scenarios_cover_the_architecture_list_and_evaluate_without_regressions() -> None:
    # S01-S12 cover the architecture list and are solved by the rules with tools; the "hard"
    # scenarios (S13+) are meant to defeat the rules and are measured separately.
    all_scenarios = load_scenarios()
    scenarios = [
        item for item in all_scenarios if not any(tag in item.scenario_id for tag in ("-hard-", "-lab-"))
    ]
    report = run_evaluation(scenarios)

    assert len(scenarios) == 12 and len(all_scenarios) >= 15
    baseline = report["metrics"]["rules"]
    tools = report["metrics"]["rules+tools"]
    vector = report["metrics"]["rules+tools+qdrant"]
    assert tools["rca_accuracy_top1"] == 1.0
    assert tools["first_action_accuracy"] == 1.0
    assert tools["rca_accuracy_top1"] > baseline["rca_accuracy_top1"]
    assert all(metrics["unsafe_recommendation_rate"] == 0 for metrics in report["metrics"].values())
    assert tools["requested_evidence_collected_rate"] == 1.0
    assert vector["rag_hit_rate"] >= 0.9
    assert report["reliability"]["idempotent"] is True
    markdown = render_markdown(report)
    assert "Précision RCA (top-1)" in markdown
    assert "S07-dependency-outage" in markdown
