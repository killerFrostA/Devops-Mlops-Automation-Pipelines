"""Vector retrieval of past incidents and runbooks with Qdrant (architecture section 6.2).

Collections follow the architecture names: ``incident_summaries`` and ``runbooks``. The same
retriever works against a Qdrant server (``compose.yaml`` profile ``intelligence``), a local
on-disk store, or an in-memory store, all through ``qdrant-client``.

``HashingEmbedder`` is a dependency-free, deterministic embedding (hashed words and word
pairs). It captures shared vocabulary, not deep semantics; any model exposing the
``Embedder`` protocol can replace it, provided the collections are re-indexed with it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from src.agents.diagnosis.contracts import KnowledgeDocument, KnowledgeRetrieval
from src.agents.diagnosis.rag_retriever import load_corpus

COLLECTIONS: dict[Literal["incident", "runbook"], str] = {
    "incident": "incident_summaries",
    "runbook": "runbooks",
}
# Hashed vectors give low cosine scores; 0.05 still drops unrelated documents.
DEFAULT_MIN_SCORE = 0.05
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-words and word-pair vectors, L2-normalized."""

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _features(self, text: str) -> Counter[str]:
        words = [word for word in _WORD.findall(text.casefold()) if len(word) > 1]
        features = Counter(words)
        features.update(
            f"{first} {second}" for first, second in zip(words, words[1:], strict=False)
        )
        return features

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for feature, count in self._features(text).items():
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign * (1.0 + math.log(count))
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


def _document_text(summary: str, tags: Sequence[str], root_cause: str | None) -> str:
    cause = root_cause.replace("_", " ") if root_cause else ""
    return f"{summary} {' '.join(tags)} {cause}"


def index_knowledge(
    client: Any,
    embedder: Embedder | None = None,
    knowledge_dir: Path | None = None,
) -> dict[str, int]:
    """(Re)create both collections and upsert every corpus document; returns counts."""
    from qdrant_client import models  # type: ignore[import-not-found, unused-ignore]

    embedder = embedder or HashingEmbedder()
    incidents, runbooks = load_corpus(knowledge_dir)
    counts = {}
    corpora: tuple[tuple[Literal["incident", "runbook"], Sequence[Any]], ...] = (
        ("incident", incidents),
        ("runbook", runbooks),
    )
    for kind, documents in corpora:
        collection = COLLECTIONS[kind]
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=embedder.dimension, distance=models.Distance.COSINE
            ),
        )
        vectors = embedder.embed(
            [_document_text(item.summary, item.tags, item.root_cause) for item in documents]
        )
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"pids-knowledge:{item.reference_id}")),
                    vector=vector,
                    payload={
                        "reference_id": item.reference_id,
                        "summary": item.summary,
                        "root_cause": item.root_cause,
                        "tags": list(item.tags),
                        "origin": item.origin,
                        "kind": kind,
                    },
                )
                for item, vector in zip(documents, vectors, strict=True)
            ],
        )
        counts[collection] = len(documents)
    return counts


class QdrantKnowledgeRetriever:
    """``KnowledgeRetriever`` backed by Qdrant vector search."""

    def __init__(
        self,
        client: Any,
        embedder: Embedder | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self._client = client
        self._embedder = embedder or HashingEmbedder()
        self._min_score = min_score

    @property
    def client(self) -> Any:
        return self._client

    @classmethod
    def in_memory(
        cls,
        knowledge_dir: Path | None = None,
        embedder: Embedder | None = None,
    ) -> QdrantKnowledgeRetriever:
        """Index the bundled corpus into an in-process Qdrant store (no server needed)."""
        from qdrant_client import QdrantClient  # type: ignore[import-not-found, unused-ignore]

        embedder = embedder or HashingEmbedder()
        client = QdrantClient(":memory:")
        index_knowledge(client, embedder, knowledge_dir)
        return cls(client, embedder)

    @classmethod
    def connect(
        cls,
        *,
        url: str | None = None,
        path: Path | None = None,
        api_key: str | None = None,
        embedder: Embedder | None = None,
    ) -> QdrantKnowledgeRetriever:
        """Use an already indexed Qdrant server (``url``) or local store (``path``)."""
        from qdrant_client import QdrantClient  # type: ignore[import-not-found, unused-ignore]

        if (url is None) == (path is None):
            raise ValueError("Provide exactly one of url or path")
        client = QdrantClient(url=url, api_key=api_key) if url else QdrantClient(path=str(path))
        return cls(client, embedder)

    def retrieve(self, query: str, limit: int = 3) -> KnowledgeRetrieval:
        if limit <= 0 or not query.strip():
            return KnowledgeRetrieval()
        vector = self._embedder.embed([query])[0]
        found: dict[str, tuple[KnowledgeDocument, ...]] = {}
        for kind, collection in COLLECTIONS.items():
            points = self._client.query_points(
                collection,
                query=vector,
                limit=limit,
                with_payload=True,
                score_threshold=self._min_score,
            ).points
            found[kind] = tuple(
                KnowledgeDocument(
                    reference_id=point.payload["reference_id"],
                    kind=kind,
                    summary=point.payload["summary"],
                    root_cause=point.payload.get("root_cause"),
                    score=round(min(max(point.score, 0.0), 1.0), 2),
                    provenance="qdrant",
                )
                for point in points
            )
        return KnowledgeRetrieval(similar_incidents=found["incident"], runbooks=found["runbook"])
