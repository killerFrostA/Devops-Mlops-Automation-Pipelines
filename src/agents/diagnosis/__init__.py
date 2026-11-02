"""Agent 2: evidence-grounded RCA; analysis only."""

from src.agents.diagnosis.contracts import (
    DiagnosisAgentResult,
    DiagnosisInput,
    DiagnosisRecommendation,
    DiagnosisResult,
    EvidenceItem,
    KnowledgeDocument,
    KnowledgeRetrieval,
    Hypothesis,
    SupplementalAnalysis,
)
from src.agents.diagnosis.evidence_collector import collect_evidence
from src.agents.diagnosis.llm_analyzer import (
    ClaudeSupplementalAnalyzer,
    DeterministicSupplementalAnalyzer,
    analyze_with_llm,
)
from src.agents.diagnosis.qdrant_retriever import QdrantKnowledgeRetriever
from src.agents.diagnosis.rag_retriever import (
    LocalKnowledgeRetriever,
    retrieve_runbooks,
    retrieve_similar_incidents,
)
from src.agents.diagnosis.result_builder import build_result, to_diagnosis_completed
from src.agents.diagnosis.rules import evaluate_hypotheses
from src.agents.diagnosis.service import DiagnosisAgent
from src.agents.diagnosis.tools import Toolbox, build_toolbox_from_env, static_toolbox

__all__ = [
    "DiagnosisAgent",
    "DiagnosisAgentResult",
    "DiagnosisInput",
    "EvidenceItem",
    "Hypothesis",
    "DiagnosisRecommendation",
    "DiagnosisResult",
    "KnowledgeDocument",
    "KnowledgeRetrieval",
    "SupplementalAnalysis",
    "collect_evidence",
    "evaluate_hypotheses",
    "retrieve_runbooks",
    "retrieve_similar_incidents",
    "LocalKnowledgeRetriever",
    "analyze_with_llm",
    "DeterministicSupplementalAnalyzer",
    "ClaudeSupplementalAnalyzer",
    "QdrantKnowledgeRetriever",
    "Toolbox",
    "build_toolbox_from_env",
    "static_toolbox",
    "build_result",
    "to_diagnosis_completed",
]
