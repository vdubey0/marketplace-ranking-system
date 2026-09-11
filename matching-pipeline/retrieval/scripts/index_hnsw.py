#!/usr/bin/env python3
"""Load cached candidate vectors into pgvector and build an HNSW index."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from retrieval.vector_store import CandidateVectorStore

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDINGS = Path(__file__).resolve().parents[1] / "artifacts/embeddings"


class CachedOnlyEmbeddings(Embeddings):
    """Fail loudly if code accidentally tries to recompute a cached vector."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("This workflow accepts cached embeddings only")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("This workflow accepts cached embeddings only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--database-url")
    parser.add_argument("--table", default="candidate_embeddings")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    database_url = args.database_url or os.getenv("RETRIEVAL_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set RETRIEVAL_DATABASE_URL or pass --database-url")
    frame = pd.read_parquet(args.embedding_dir / "candidates.parquet").sort_values("entity_id")
    dimensions = set(frame.dimensions.astype(int))
    if len(dimensions) != 1:
        raise ValueError("Candidate artifact must contain one vector dimension")
    store = CandidateVectorStore(database_url, CachedOnlyEmbeddings(), dimensions.pop(), args.table)
    try:
        store.initialize(overwrite=args.overwrite)
        store.add_cached_embeddings(
            frame.entity_id.astype(str).tolist(),
            [list(vector) for vector in frame.embedding],
            frame.content_sha256.astype(str).tolist(),
        )
        store.build_hnsw(m=args.m, ef_construction=args.ef_construction)
    finally:
        store.close()
    print(f"Indexed {len(frame):,} cached candidate vectors in {args.table}")


if __name__ == "__main__":
    main()
