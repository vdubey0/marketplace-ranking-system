#!/usr/bin/env python3
"""Retrieve a shortlist for one stored job, then rerank it with LightGBM."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from modeling.reranker import rerank_candidates
from retrieval.embeddings import embedding_matrix
from retrieval.exact_search import exact_cosine_search

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ground-truth-generation/data/v1"
EMBEDDINGS = ROOT / "matching-pipeline/retrieval/artifacts/embeddings"
MODEL = ROOT / "matching-pipeline/modeling/artifacts/model/performance_model.joblib"


def exact_retrieval(job_vector, candidate_embeddings: pd.DataFrame, k: int) -> pd.DataFrame:
    indices, scores = exact_cosine_search(
        np.asarray(job_vector, dtype=np.float32).reshape(1, -1),
        embedding_matrix(candidate_embeddings), k,
    )
    return pd.DataFrame({
        "candidate_id": candidate_embeddings.iloc[indices[0]].entity_id.to_numpy(),
        "retrieval_rank": np.arange(1, len(indices[0]) + 1),
        "embedding_similarity": scores[0],
    })


def hnsw_retrieval(job_vector, candidate_embeddings: pd.DataFrame, k: int,
                   database_url: str, table: str) -> pd.DataFrame:
    from index_hnsw import CachedOnlyEmbeddings
    from retrieval.vector_store import CandidateVectorStore

    dimensions = int(candidate_embeddings.dimensions.iloc[0])
    store = CandidateVectorStore(database_url, CachedOnlyEmbeddings(), dimensions, table)
    store.connect()
    try:
        matches = store.search_by_vector(list(job_vector), k)
    finally:
        store.close()
    return pd.DataFrame({
        "candidate_id": [match.candidate_id for match in matches],
        "retrieval_rank": [match.rank for match in matches],
        "embedding_similarity": [1.0 - match.distance for match in matches],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", choices=("exact", "hnsw"), default="exact")
    parser.add_argument("--database-url")
    parser.add_argument("--table", default="candidate_embeddings")
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    if args.retrieve_k < args.top_k or args.top_k <= 0:
        raise ValueError("retrieve-k must be at least top-k, and both must be positive")

    jobs = pd.read_parquet(DATA / "jobs.parquet")
    matching_jobs = jobs.loc[jobs.job_id == args.job_id]
    if len(matching_jobs) != 1:
        raise ValueError(f"unknown job_id: {args.job_id}")
    job = matching_jobs.iloc[0]
    candidate_embeddings = pd.read_parquet(EMBEDDINGS / "candidates.parquet").sort_values(
        "entity_id"
    ).reset_index(drop=True)
    job_embeddings = pd.read_parquet(EMBEDDINGS / "jobs.parquet").set_index("entity_id")
    job_vector = job_embeddings.loc[args.job_id, "embedding"]

    if args.backend == "exact":
        retrieved = exact_retrieval(job_vector, candidate_embeddings, args.retrieve_k)
    else:
        load_dotenv(ROOT / ".env")
        database_url = args.database_url or os.getenv("RETRIEVAL_DATABASE_URL")
        if not database_url:
            raise SystemExit("Set RETRIEVAL_DATABASE_URL or pass --database-url")
        retrieved = hnsw_retrieval(
            job_vector, candidate_embeddings, args.retrieve_k, database_url, args.table
        )

    candidates = pd.read_parquet(DATA / "candidates.parquet")
    assessments = pd.read_parquet(DATA / "candidate_assessments.parquet")
    ranked = rerank_candidates(
        job, retrieved, candidates, assessments, joblib.load(args.model), top_k=args.top_k
    )
    print(f"{job.title} ({job.job_id})")
    print(ranked.to_string(index=False, formatters={
        "embedding_similarity": "{:.3f}".format,
        "performance_probability": "{:.3f}".format,
    }))


if __name__ == "__main__":
    main()
