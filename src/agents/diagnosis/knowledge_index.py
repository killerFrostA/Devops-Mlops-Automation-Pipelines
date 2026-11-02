"""Index Agent 2's knowledge corpus into Qdrant and optionally try a search.

    py -3 -m src.agents.diagnosis.knowledge_index --qdrant-url http://localhost:6333
    py -3 -m src.agents.diagnosis.knowledge_index --qdrant-path .qdrant --query "oom memory limit"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agents.diagnosis.qdrant_retriever import QdrantKnowledgeRetriever, index_knowledge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index the diagnosis knowledge corpus into Qdrant.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--qdrant-url", help="Qdrant server URL, e.g. http://localhost:6333.")
    target.add_argument("--qdrant-path", type=Path, help="Local on-disk Qdrant store directory.")
    parser.add_argument("--knowledge-dir", type=Path, help="Corpus directory (default: bundled).")
    parser.add_argument("--query", help="Run one search after indexing.")
    args = parser.parse_args(argv)

    try:
        retriever = QdrantKnowledgeRetriever.connect(url=args.qdrant_url, path=args.qdrant_path)
        counts = index_knowledge(retriever.client, knowledge_dir=args.knowledge_dir)
    except Exception as exc:  # Connection, corpus and Qdrant errors are all reported.
        print(f"Indexing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for collection, count in counts.items():
        print(f"{collection}: {count} documents indexed")
    if args.query:
        result = retriever.retrieve(args.query, 3)
        for item in (*result.similar_incidents, *result.runbooks):
            print(f"- {item.kind} {item.reference_id} ({item.root_cause}) score {item.score:.2f}: {item.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
