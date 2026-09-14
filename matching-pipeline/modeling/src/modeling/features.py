"""Explainable pre-outcome features for candidate-job pairs."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SENIORITY = {"entry": 0, "mid": 1, "senior": 2, "leadership": 3}

BASE_FEATURE_COLUMNS = [
    "interview_score",
    "assessment_score",
    "embedding_similarity",
    "required_skill_coverage",
    "preferred_skill_coverage",
    "role_family_match",
    "role_type_match",
    "role_track_match",
    "seniority_gap",
    "absolute_seniority_gap",
    "candidate_years_experience",
    "minimum_years_experience",
    "experience_gap",
    "meets_experience_minimum",
    "domain_match",
    "required_certification_coverage",
    "missing_required_certification",
    "required_skill_count",
    "preferred_skill_count",
]
PCA_COMPONENTS = 64
PCA_DIFFERENCE_COLUMNS = [f"pca_abs_diff_{number:02d}" for number in range(PCA_COMPONENTS)]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + PCA_DIFFERENCE_COLUMNS

FORBIDDEN_FEATURES = {
    "name", "fit_profile", "is_synthetic", "relevance_grade", "relevance_score",
    "skill_score", "occupation_score", "seniority_score", "experience_score",
    "domain_score", "hard_gate_failed", "true_success_probability", "potential_success",
    "selected", "observed_success", "latent_ability", "latent_reliability",
    "latent_domain_depth", "latent_skill_mastery", "latent_difficulty",
    "latent_specialization",
}


def _set(value: Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    return {str(item).strip().casefold() for item in value if str(item).strip()}


def _coverage(candidate_values, required_values, *, empty_value: float) -> float:
    required = _set(required_values)
    return len(_set(candidate_values) & required) / len(required) if required else empty_value


def build_pair_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Create numeric features from a frame containing joined candidate/job fields."""
    rows = []
    for row in pairs.itertuples(index=False):
        candidate_level = SENIORITY.get(row.candidate_seniority, 1)
        job_level = SENIORITY.get(row.job_seniority, 1)
        seniority_gap = candidate_level - job_level
        candidate_years = float(row.candidate_years_experience)
        minimum_years = (float(row.minimum_years_experience)
                         if pd.notna(row.minimum_years_experience) else 0.0)
        required_certifications = _set(row.required_certifications)
        certification_coverage = _coverage(
            row.candidate_certifications, row.required_certifications, empty_value=1.0
        )
        rows.append({
            "interview_score": float(row.interview_score),
            "assessment_score": float(row.assessment_score),
            "embedding_similarity": float(row.embedding_similarity),
            "required_skill_coverage": _coverage(row.candidate_skills, row.required_skills, empty_value=1.0),
            "preferred_skill_coverage": _coverage(row.candidate_skills, row.preferred_skills, empty_value=0.0),
            "role_family_match": float(row.candidate_role_family == row.job_role_family),
            "role_type_match": float(row.candidate_role_type == row.job_role_type),
            "role_track_match": float(row.candidate_role_track == row.job_role_track),
            "seniority_gap": float(seniority_gap),
            "absolute_seniority_gap": float(abs(seniority_gap)),
            "candidate_years_experience": candidate_years,
            "minimum_years_experience": minimum_years,
            "experience_gap": candidate_years - minimum_years,
            "meets_experience_minimum": float(candidate_years >= minimum_years),
            "domain_match": float(row.candidate_domain == row.job_domain and row.job_domain != "unknown"),
            "required_certification_coverage": certification_coverage,
            "missing_required_certification": float(bool(required_certifications) and certification_coverage < 1.0),
            "required_skill_count": float(len(_set(row.required_skills))),
            "preferred_skill_count": float(len(_set(row.preferred_skills))),
        })
    return pd.DataFrame(rows, columns=BASE_FEATURE_COLUMNS, dtype=np.float32)
