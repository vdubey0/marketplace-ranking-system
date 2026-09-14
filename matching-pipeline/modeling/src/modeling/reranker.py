"""Join retrieval results to explainable features and rerank them with a model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import BASE_FEATURE_COLUMNS, build_pair_features


def rerank_candidates(
    job: pd.Series,
    retrieved: pd.DataFrame,
    candidates: pd.DataFrame,
    assessments: pd.DataFrame,
    model,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return retrieved candidates ordered by predicted success probability."""
    required = {"candidate_id", "retrieval_rank", "embedding_similarity"}
    missing = required - set(retrieved.columns)
    if missing:
        raise ValueError(f"retrieved candidates are missing columns: {sorted(missing)}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if retrieved.candidate_id.duplicated().any():
        raise ValueError("retrieved candidate IDs must be unique")

    candidate_fields = candidates.rename(columns={
        "skills": "candidate_skills",
        "years_experience": "candidate_years_experience",
        "domain": "candidate_domain",
        "role_family": "candidate_role_family",
        "role_type": "candidate_role_type",
        "role_track": "candidate_role_track",
        "seniority": "candidate_seniority",
        "certifications": "candidate_certifications",
    })
    pair_rows = retrieved.merge(candidate_fields, on="candidate_id", validate="one_to_one")
    pair_rows = pair_rows.merge(
        assessments[["candidate_id", "interview_score", "assessment_score"]],
        on="candidate_id", validate="one_to_one",
    )
    if len(pair_rows) != len(retrieved):
        found = set(pair_rows.candidate_id)
        raise ValueError(f"missing candidate data for: {sorted(set(retrieved.candidate_id) - found)}")

    job_values = {
        "job_id": job.job_id,
        "minimum_years_experience": job.minimum_years_experience,
        "required_skills": job.required_skills,
        "preferred_skills": job.preferred_skills,
        "required_certifications": job.required_certifications,
        "job_domain": job.domain,
        "job_role_family": job.role_family,
        "job_role_type": job.role_type,
        "job_role_track": job.role_track,
        "job_seniority": job.seniority,
    }
    for name, value in job_values.items():
        pair_rows[name] = [value] * len(pair_rows)

    features = build_pair_features(pair_rows)
    probabilities = np.asarray(model.predict_proba(features[BASE_FEATURE_COLUMNS]))[:, 1]
    result = pair_rows[["candidate_id", "name", "current_title", "retrieval_rank",
                        "embedding_similarity"]].copy()
    result["performance_probability"] = probabilities
    result = result.sort_values(
        ["performance_probability", "embedding_similarity", "candidate_id"],
        ascending=[False, False, True],
    ).head(top_k).reset_index(drop=True)
    result.insert(0, "final_rank", np.arange(1, len(result) + 1))
    return result
