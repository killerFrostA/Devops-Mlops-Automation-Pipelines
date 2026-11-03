from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.agents.diagnosis.contracts import KnowledgeDocument, KnowledgeRetrieval


class _CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    tags: tuple[str, ...] = Field(min_length=1)
    root_cause: str | None
    origin: Literal["synthetic_example", "repository_source", "lab_observation"]

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        if any(not tag.strip() for tag in tags):
            raise ValueError("Knowledge tags must not be empty")
        if len({tag.casefold() for tag in tags}) != len(tags):
            raise ValueError("Knowledge tags must be unique, ignoring case")
        return tags


class _Corpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1]
    documents: tuple[_CorpusDocument, ...]


_DEFAULT_KNOWLEDGE_DIR = Path(__file__).with_name("knowledge")
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _read_corpus(path: Path, kind: Literal["incident", "runbook"]) -> tuple[_CorpusDocument, ...]:
    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to read {kind} knowledge corpus at {path}: {exc}") from exc

    try:
        raw_corpus = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {kind} knowledge corpus at {path} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    try:
        corpus = _Corpus.model_validate(raw_corpus)
    except ValidationError as exc:
        raise ValueError(f"Invalid {kind} knowledge corpus at {path}: {exc}") from exc

    reference_ids = [document.reference_id for document in corpus.documents]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError(f"Duplicate reference_id in {kind} knowledge corpus at {path}.")

    if kind == "incident" and any(document.root_cause is None for document in corpus.documents):
        raise ValueError(f"Every incident in {path} must specify root_cause.")
    if kind == "runbook" and any(document.root_cause is not None for document in corpus.documents):
        raise ValueError(f"Runbooks in {path} must set root_cause to null.")

    return corpus.documents


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.casefold()))


def _rank_documents(
    query: str,
    documents: tuple[_CorpusDocument, ...],
    kind: Literal["incident", "runbook"],
    limit: int,
) -> tuple[KnowledgeDocument, ...]:
    query_tokens = _tokens(query)
    if limit <= 0 or not query_tokens:
        return ()

    ranked: list[KnowledgeDocument] = []
    for document in documents:
        matched_tags = sum(bool(_tokens(tag) & query_tokens) for tag in document.tags)
        if matched_tags == 0:
            continue
        ranked.append(
            KnowledgeDocument(
                reference_id=document.reference_id,
                kind=kind,
                summary=document.summary,
                root_cause=document.root_cause,
                score=round(matched_tags / len(document.tags), 2),
            )
        )

    return tuple(sorted(ranked, key=lambda item: (-item.score, item.reference_id))[:limit])


def retrieve_similar_incidents(query: str, limit: int = 3) -> list[KnowledgeDocument]:
    """Retrieve incidents from the validated, bundled local JSON corpus."""
    incidents = _read_corpus(_DEFAULT_KNOWLEDGE_DIR / "incidents.json", "incident")
    return list(_rank_documents(query, incidents, "incident", limit))


def retrieve_runbooks(query: str, limit: int = 3) -> list[KnowledgeDocument]:
    """Retrieve runbooks from the validated, bundled local JSON corpus."""
    runbooks = _read_corpus(_DEFAULT_KNOWLEDGE_DIR / "runbooks.json", "runbook")
    return list(_rank_documents(query, runbooks, "runbook", limit))


def load_corpus(
    knowledge_dir: Path | None = None,
) -> tuple[tuple[_CorpusDocument, ...], tuple[_CorpusDocument, ...]]:
    """Load ``incidents.json``/``runbooks.json`` plus any ``incidents_*``/``runbooks_*`` files."""
    corpus_dir = knowledge_dir or _DEFAULT_KNOWLEDGE_DIR
    loaded: dict[str, list[_CorpusDocument]] = {}
    kinds: tuple[tuple[Literal["incident", "runbook"], str], ...] = (
        ("incident", "incidents"),
        ("runbook", "runbooks"),
    )
    for kind, stem in kinds:
        paths = [corpus_dir / f"{stem}.json", *sorted(corpus_dir.glob(f"{stem}_*.json"))]
        loaded[kind] = [document for path in paths for document in _read_corpus(path, kind)]

    all_ids = [document.reference_id for documents in loaded.values() for document in documents]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"Knowledge reference_id values must be unique across {corpus_dir}.")
    return tuple(loaded["incident"]), tuple(loaded["runbook"])


class LocalKnowledgeRetriever:
    """Search local JSON knowledge; results are reference context, not incident evidence."""

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self._incidents, self._runbooks = load_corpus(knowledge_dir)

    def retrieve(self, query: str, limit: int = 3) -> KnowledgeRetrieval:
        return KnowledgeRetrieval(
            similar_incidents=_rank_documents(query, self._incidents, "incident", limit),
            runbooks=_rank_documents(query, self._runbooks, "runbook", limit),
        )
