#!/usr/bin/env python3
"""Build a similar-job scenario and compare greedy with optimal allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from allocation.optimizer import greedy_allocate, optimize_allocation
from allocation.scenarios import (SIMILARITY_FIELDS, has_required_certifications,
                                  select_similar_jobs)
from modeling.reranker import rerank_candidates
from retrieval.embeddings import embedding_matrix
from retrieval.exact_search import exact_cosine_search

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "ground-truth-generation/data/v1"
EMBEDDINGS = ROOT / "matching-pipeline/retrieval/artifacts/embeddings"
MODEL = ROOT / "matching-pipeline/modeling/artifacts/model/performance_model.joblib"
DEFAULT_OUTPUT = ROOT / "matching-pipeline/allocation/artifacts/demo"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--seed-job-id")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.jobs <= 1 or args.retrieve_k < args.jobs:
        raise ValueError("use at least two jobs and retrieve at least as many candidates as jobs")

    jobs = pd.read_parquet(DATA / "jobs.parquet")
    candidates = pd.read_parquet(DATA / "candidates.parquet")
    assessments = pd.read_parquet(DATA / "candidate_assessments.parquet")
    candidate_embeddings = pd.read_parquet(EMBEDDINGS / "candidates.parquet").sort_values(
        "entity_id"
    ).reset_index(drop=True)
    job_embeddings = pd.read_parquet(EMBEDDINGS / "jobs.parquet")
    selected_jobs = select_similar_jobs(jobs, job_embeddings, args.jobs, args.seed_job_id)
    vectors = job_embeddings.set_index("entity_id")
    candidate_matrix = embedding_matrix(candidate_embeddings)
    fitted_model = joblib.load(args.model)

    pair_frames = []
    for job in selected_jobs.itertuples(index=False):
        job_vector = np.asarray(vectors.loc[job.job_id, "embedding"], dtype=np.float32)
        indices, scores = exact_cosine_search(job_vector.reshape(1, -1), candidate_matrix,
                                              args.retrieve_k)
        retrieved = pd.DataFrame({
            "candidate_id": candidate_embeddings.iloc[indices[0]].entity_id.to_numpy(),
            "retrieval_rank": np.arange(1, len(indices[0]) + 1),
            "embedding_similarity": scores[0],
        })
        scored = rerank_candidates(
            pd.Series(job._asdict()), retrieved, candidates, assessments,
            fitted_model, top_k=args.retrieve_k,
        )
        certification_by_id = candidates.set_index("candidate_id").certifications
        scored["eligible"] = scored.candidate_id.map(
            lambda candidate_id: has_required_certifications(
                certification_by_id.loc[candidate_id], job.required_certifications
            )
        )
        scored.insert(0, "job_id", job.job_id)
        scored.insert(1, "job_title", job.title)
        pair_frames.append(scored)
    pairs = pd.concat(pair_frames, ignore_index=True)

    optimal = optimize_allocation(pairs)
    greedy = greedy_allocate(pairs, require_all=False)
    optimal_total = float(optimal.performance_probability.sum())
    greedy_total = float(greedy.performance_probability.sum())
    candidate_job_counts = pairs.loc[pairs.eligible].groupby("candidate_id").job_id.nunique()
    independent_choices = pairs.loc[pairs.eligible].sort_values(
        ["job_id", "performance_probability"], ascending=[True, False]
    ).groupby("job_id").head(1)
    report = {
        "jobs": len(selected_jobs),
        "filled_jobs": len(optimal),
        "unfilled_jobs": sorted(set(selected_jobs.job_id) - set(optimal.job_id)),
        "retrieved_per_job": args.retrieve_k,
        "job_profile": {field: str(selected_jobs.iloc[0][field]) for field in SIMILARITY_FIELDS},
        "candidates_competing_for_multiple_jobs": int((candidate_job_counts > 1).sum()),
        "independent_top_choice_conflicts": int(
            (independent_choices.candidate_id.value_counts() > 1).sum()
        ),
        "greedy_total_expected_successes": greedy_total,
        "optimal_total_expected_successes": optimal_total,
        "optimization_gain": optimal_total - greedy_total,
        "solver": "scipy.optimize.milp",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_jobs.to_parquet(args.output_dir / "similar_jobs.parquet", index=False)
    pairs.to_parquet(args.output_dir / "scored_pairs.parquet", index=False)
    greedy.to_csv(args.output_dir / "greedy_assignments.csv", index=False)
    optimal.to_csv(args.output_dir / "optimal_assignments.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nOptimal assignments")
    print(optimal[["job_title", "candidate_id", "name", "performance_probability"]]
          .to_string(index=False, formatters={"performance_probability": "{:.3f}".format}))


if __name__ == "__main__":
    main()
