#!/usr/bin/env python3
"""Run exact cosine retrieval and write ranks plus Recall@K reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from retrieval.embeddings import embedding_matrix
from retrieval.evaluation import evaluate_retrieval
from retrieval.exact_search import exact_cosine_search

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "ground-truth-generation/data/v1"
DEFAULT_EMBEDDINGS = Path(__file__).resolve().parents[1] / "artifacts/embeddings"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/exact"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 25, 50, 100, 250, 500])
    parser.add_argument("--minimum-grade", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    candidates = pd.read_parquet(args.embedding_dir / "candidates.parquet").sort_values("entity_id")
    jobs = pd.read_parquet(args.embedding_dir / "jobs.parquet").sort_values("entity_id")
    if candidates.model.nunique() != 1 or jobs.model.nunique() != 1:
        raise ValueError("Each artifact must contain one embedding model")
    if candidates.model.iloc[0] != jobs.model.iloc[0]:
        raise ValueError("Candidate and job embedding models differ")
    indices, scores = exact_cosine_search(
        embedding_matrix(jobs), embedding_matrix(candidates), max(args.ks)
    )
    rows = []
    candidate_ids = candidates.entity_id.to_numpy()
    for job_row, job_id in enumerate(jobs.entity_id):
        for rank, (candidate_row, score) in enumerate(
            zip(indices[job_row], scores[job_row], strict=True), start=1
        ):
            rows.append({"job_id": job_id, "candidate_id": candidate_ids[candidate_row],
                         "rank": rank, "cosine_similarity": float(score)})
    retrieval = pd.DataFrame(rows)
    relevance = pd.read_parquet(
        args.data_dir / "relevance_ground_truth.parquet",
        columns=["job_id", "candidate_id", "relevance_grade"],
        filters=[("job_id", "in", jobs.entity_id.tolist())],
    )
    # In a partial-candidate pilot, evaluate only positives that were searchable.
    relevance = relevance.loc[relevance.candidate_id.isin(set(candidate_ids))]
    per_job, summary = evaluate_retrieval(
        retrieval, relevance, args.ks, minimum_grade=args.minimum_grade
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval.to_parquet(args.output_dir / "retrieval.parquet", index=False)
    per_job.to_parquet(args.output_dir / "per_job_metrics.parquet", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
