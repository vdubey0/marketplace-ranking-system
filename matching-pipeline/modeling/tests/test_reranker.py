import numpy as np
import pandas as pd

from modeling.reranker import rerank_candidates


class ScoreFirstFeature:
    def predict_proba(self, features):
        probability = features.interview_score.to_numpy()
        return np.column_stack([1 - probability, probability])


def test_reranker_preserves_retrieval_information_and_orders_by_model_score():
    job = pd.Series({
        "job_id": "j1", "minimum_years_experience": 2, "required_skills": ["python"],
        "preferred_skills": [], "required_certifications": [], "domain": "ml",
        "role_family": "engineering", "role_type": "individual_contributor",
        "role_track": "technical", "seniority": "mid",
    })
    retrieved = pd.DataFrame({
        "candidate_id": ["c1", "c2"], "retrieval_rank": [1, 2],
        "embedding_similarity": [0.9, 0.8],
    })
    candidates = pd.DataFrame({
        "candidate_id": ["c1", "c2"], "name": ["One", "Two"],
        "current_title": ["Engineer", "Engineer"], "skills": [["python"], ["python"]],
        "years_experience": [3, 3], "domain": ["ml", "ml"],
        "role_family": ["engineering", "engineering"],
        "role_type": ["individual_contributor", "individual_contributor"],
        "role_track": ["technical", "technical"], "seniority": ["mid", "mid"],
        "certifications": [[], []],
    })
    assessments = pd.DataFrame({
        "candidate_id": ["c1", "c2"], "interview_score": [0.2, 0.8],
        "assessment_score": [0.5, 0.5],
    })

    result = rerank_candidates(job, retrieved, candidates, assessments, ScoreFirstFeature())

    assert result.candidate_id.tolist() == ["c2", "c1"]
    assert result.retrieval_rank.tolist() == [2, 1]
    assert result.final_rank.tolist() == [1, 2]
