"""Helpers for creating understandable, competing-job allocation scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

SIMILARITY_FIELDS = ["role_family", "role_type", "role_track", "seniority", "domain"]


def select_similar_jobs(jobs: pd.DataFrame, job_embeddings: pd.DataFrame,
                        count: int, seed_job_id: str | None = None) -> pd.DataFrame:
    """Choose semantically close jobs sharing the same structured job profile."""
    if seed_job_id is not None:
        match = jobs.loc[jobs.job_id == seed_job_id]
        if len(match) != 1:
            raise ValueError(f"unknown job_id: {seed_job_id}")
        anchor = match.iloc[0]
        pool = jobs.copy()
        for field in SIMILARITY_FIELDS:
            pool = pool.loc[pool[field] == anchor[field]]
    else:
        groups = jobs.groupby(SIMILARITY_FIELDS, dropna=False).size()
        valid = groups.loc[groups >= count]
        if valid.empty:
            raise ValueError(f"no job family contains {count} similar jobs")
        signature = valid.sort_values(ascending=False).index[0]
        pool = jobs.copy()
        for field, value in zip(SIMILARITY_FIELDS, signature):
            pool = pool.loc[pool[field].isna() if pd.isna(value) else pool[field] == value]
        anchor = pool.sort_values("job_id").iloc[0]
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} jobs share the requested job profile")

    vectors = job_embeddings.set_index("entity_id")
    anchor_vector = np.asarray(vectors.loc[anchor.job_id, "embedding"], dtype=np.float32)
    pool_matrix = np.stack(vectors.loc[pool.job_id, "embedding"].map(
        lambda value: np.asarray(value, dtype=np.float32)
    ))
    similarities = pool_matrix @ anchor_vector / (
        np.linalg.norm(pool_matrix, axis=1) * np.linalg.norm(anchor_vector)
    )
    return pool.assign(job_similarity=similarities).sort_values(
        ["job_similarity", "job_id"], ascending=[False, True]
    ).head(count).reset_index(drop=True)


def has_required_certifications(candidate_certifications, required_certifications) -> bool:
    candidate = {str(value).strip().casefold() for value in candidate_certifications}
    required = {str(value).strip().casefold() for value in required_certifications}
    return required.issubset(candidate)
