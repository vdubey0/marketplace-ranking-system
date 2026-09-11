#!/usr/bin/env python3
"""Compare HNSW serving results with the exact cosine reference."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from index_hnsw import CachedOnlyEmbeddings
from retrieval.vector_store import CandidateVectorStore

ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, default=RETRIEVAL_ROOT / "artifacts/embeddings")
    parser.add_argument("--exact-dir", type=Path, default=RETRIEVAL_ROOT / "artifacts/exact")
    parser.add_argument("--output-dir", type=Path, default=RETRIEVAL_ROOT / "artifacts/hnsw")
    parser.add_argument("--database-url")
    parser.add_argument("--table", default="candidate_embeddings")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 25, 50, 100])
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    database_url = args.database_url or os.getenv("RETRIEVAL_DATABASE_URL")
    if not database_url:
        raise SystemExit("Set RETRIEVAL_DATABASE_URL or pass --database-url")
    jobs = pd.read_parquet(args.embedding_dir / "jobs.parquet").sort_values("entity_id")
    exact = pd.read_parquet(args.exact_dir / "retrieval.parquet")
    dimensions = int(jobs.dimensions.iloc[0])
    store = CandidateVectorStore(database_url, CachedOnlyEmbeddings(), dimensions, args.table)
    store.connect()
    rows = []
    try:
        for job in jobs.itertuples(index=False):
            start = time.perf_counter()
            matches = store.search_by_vector(list(job.embedding), max(args.ks))
            elapsed_ms = (time.perf_counter() - start) * 1000
            rows.extend({"job_id": job.entity_id, "candidate_id": match.candidate_id,
                         "rank": match.rank, "distance": match.distance,
                         "latency_ms": elapsed_ms} for match in matches)
    finally:
        store.close()
    hnsw = pd.DataFrame(rows)
    per_job = []
    for job_id, group in hnsw.groupby("job_id"):
        exact_job = exact.loc[exact.job_id == job_id]
        row = {"job_id": job_id, "latency_ms": float(group.latency_ms.iloc[0])}
        for k in sorted(set(args.ks)):
            approximate_ids = set(group.loc[group["rank"] <= k, "candidate_id"])
            exact_ids = set(exact_job.loc[exact_job["rank"] <= k, "candidate_id"])
            row[f"overlap_at_{k}"] = len(approximate_ids & exact_ids) / len(exact_ids) if exact_ids else None
        per_job.append(row)
    metrics = pd.DataFrame(per_job)
    summary = {
        "jobs": len(metrics),
        "latency_ms": {"p50": float(metrics.latency_ms.median()),
                       "p95": float(metrics.latency_ms.quantile(.95))},
        "mean_exact_overlap": {str(k): float(metrics[f"overlap_at_{k}"].mean()) for k in args.ks},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hnsw.to_parquet(args.output_dir / "retrieval.parquet", index=False)
    metrics.to_parquet(args.output_dir / "per_job_metrics.parquet", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
