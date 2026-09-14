#!/usr/bin/env python3
"""Build a bounded, reproducible candidate-job performance dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from sklearn.decomposition import PCA

from modeling.features import FEATURE_COLUMNS, PCA_COMPONENTS, PCA_DIFFERENCE_COLUMNS, build_pair_features

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "ground-truth-generation/data/v1"
DEFAULT_EMBEDDINGS = ROOT / "matching-pipeline/retrieval/artifacts/embeddings"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=2000)
    parser.add_argument("--pairs-per-job", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if args.jobs <= 0 or args.pairs_per_job <= 0:
        raise ValueError("jobs and pairs-per-job must be positive")

    jobs = pd.read_parquet(args.data_dir / "jobs.parquet")
    selected_jobs = jobs.sample(n=min(args.jobs, len(jobs)), random_state=args.seed)
    selected_ids = set(selected_jobs.job_id)
    chunks = []
    columns = ["job_id", "candidate_id", "relevance_grade", "successful_performance"]
    for batch in pq.ParquetFile(args.data_dir / "marketplace_outcomes.parquet").iter_batches(
        batch_size=500_000, columns=columns
    ):
        frame = batch.to_pandas()
        frame = frame.loc[frame.job_id.isin(selected_ids)]
        if len(frame):
            chunks.append(frame)
    outcomes = pd.concat(chunks, ignore_index=True)
    # Hash ordering is stable and avoids using the target to select examples.
    outcomes["sample_priority"] = pd.util.hash_pandas_object(
        outcomes.job_id + "|" + outcomes.candidate_id + f"|{args.seed}", index=False
    )
    ordered = outcomes.sort_values(["job_id", "sample_priority"])
    hard_negatives = ordered.loc[ordered.relevance_grade == 0]
    other = ordered.loc[ordered.relevance_grade != 0].copy()
    negative_counts = hard_negatives.groupby("job_id").size()
    other["within_job_rank"] = other.groupby("job_id").cumcount()
    capacity = other.job_id.map(lambda job_id: max(0, args.pairs_per_job - negative_counts.get(job_id, 0)))
    other = other.loc[other.within_job_rank < capacity].drop(columns="within_job_rank")
    outcomes = (pd.concat([hard_negatives, other], ignore_index=True)
                .drop(columns="sample_priority").sort_values(["job_id", "candidate_id"])
                .reset_index(drop=True))

    candidates = pd.read_parquet(args.data_dir / "candidates.parquet").rename(columns={
        "skills": "candidate_skills", "years_experience": "candidate_years_experience",
        "domain": "candidate_domain", "role_family": "candidate_role_family",
        "role_type": "candidate_role_type", "role_track": "candidate_role_track",
        "seniority": "candidate_seniority", "certifications": "candidate_certifications",
    })
    assessments = pd.read_parquet(
        args.data_dir / "candidate_assessments.parquet",
        columns=["candidate_id", "interview_score", "assessment_score"],
    )
    candidates = candidates.merge(assessments, on="candidate_id", validate="one_to_one")
    selected_jobs = selected_jobs.rename(columns={
        "required_skills": "required_skills", "preferred_skills": "preferred_skills",
        "domain": "job_domain", "role_family": "job_role_family", "role_type": "job_role_type",
        "role_track": "job_role_track", "seniority": "job_seniority",
    })
    candidate_columns = ["candidate_id", "interview_score", "assessment_score",
                         "candidate_skills", "candidate_years_experience",
                         "candidate_domain", "candidate_role_family", "candidate_role_type",
                         "candidate_role_track", "candidate_seniority", "candidate_certifications"]
    job_columns = ["job_id", "minimum_years_experience", "required_skills", "preferred_skills",
                   "required_certifications", "job_domain", "job_role_family", "job_role_type",
                   "job_role_track", "job_seniority"]
    joined = outcomes.merge(candidates[candidate_columns], on="candidate_id", validate="many_to_one")
    joined = joined.merge(selected_jobs[job_columns], on="job_id", validate="many_to_one")

    candidate_embeddings = pd.read_parquet(
        args.embedding_dir / "candidates.parquet", columns=["entity_id", "embedding"]
    ).set_index("entity_id")
    job_embeddings = pd.read_parquet(
        args.embedding_dir / "jobs.parquet", columns=["entity_id", "embedding"]
    ).set_index("entity_id")
    candidate_matrix = np.stack(candidate_embeddings.embedding.map(
        lambda value: np.asarray(value, dtype=np.float32)
    ))
    candidate_matrix /= np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=args.seed)
    candidate_pca = pca.fit_transform(candidate_matrix).astype(np.float32)
    candidate_pca_by_id = pd.DataFrame(candidate_pca, index=candidate_embeddings.index)

    selected_job_ids = pd.Index(joined.job_id.unique())
    selected_job_matrix = np.stack(job_embeddings.loc[selected_job_ids, "embedding"].map(
        lambda value: np.asarray(value, dtype=np.float32)
    ))
    selected_job_matrix /= np.linalg.norm(selected_job_matrix, axis=1, keepdims=True)
    job_pca_by_id = pd.DataFrame(
        pca.transform(selected_job_matrix).astype(np.float32), index=selected_job_ids
    )
    similarities = np.empty(len(joined), dtype=np.float32)
    # Work in bounded chunks instead of materializing two 400k x 1536 matrices.
    for start in range(0, len(joined), 20_000):
        stop = min(start + 20_000, len(joined))
        part = joined.iloc[start:stop]
        c_vectors = np.stack(candidate_embeddings.loc[part.candidate_id, "embedding"].map(
            lambda value: np.asarray(value, dtype=np.float32)
        ))
        j_vectors = np.stack(job_embeddings.loc[part.job_id, "embedding"].map(
            lambda value: np.asarray(value, dtype=np.float32)
        ))
        similarities[start:stop] = np.einsum("ij,ij->i", c_vectors, j_vectors) / (
            np.linalg.norm(c_vectors, axis=1) * np.linalg.norm(j_vectors, axis=1)
        )
    joined["embedding_similarity"] = similarities
    features = build_pair_features(joined)
    candidate_pair_pca = candidate_pca_by_id.loc[joined.candidate_id].to_numpy()
    job_pair_pca = job_pca_by_id.loc[joined.job_id].to_numpy()
    pca_differences = pd.DataFrame(
        np.abs(candidate_pair_pca - job_pair_pca), columns=PCA_DIFFERENCE_COLUMNS
    )
    features = pd.concat([features, pca_differences], axis=1)[FEATURE_COLUMNS]
    dataset = pd.concat([
        joined[["job_id", "candidate_id", "successful_performance"]].reset_index(drop=True), features
    ], axis=1)
    dataset["successful_performance"] = dataset.successful_performance.astype("int8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output_dir / "performance_pairs.parquet", index=False)
    joblib.dump(pca, args.output_dir / "embedding_pca.joblib")
    manifest = {
        "seed": args.seed, "rows": len(dataset), "jobs": int(dataset.job_id.nunique()),
        "candidates": int(dataset.candidate_id.nunique()),
        "positive_rate": float(dataset.successful_performance.mean()),
        "pairs_per_job_max": args.pairs_per_job, "features": FEATURE_COLUMNS,
        "sampling_uses_target": False, "all_available_grade_zero_pairs_retained": True,
        "pca_components": PCA_COMPONENTS,
        "pca_fit_source": "normalized candidate embeddings only",
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
