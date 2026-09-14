import unittest

import pandas as pd

from modeling.features import BASE_FEATURE_COLUMNS, FORBIDDEN_FEATURES, build_pair_features


class FeatureTests(unittest.TestCase):
    def test_features_are_explainable_and_leakage_safe(self):
        pair = pd.DataFrame([{
            "interview_score": .75, "assessment_score": .80,
            "embedding_similarity": .8,
            "candidate_skills": ["Python", "SQL"], "required_skills": ["Python", "PyTorch"],
            "preferred_skills": ["SQL"], "candidate_role_family": "engineering",
            "job_role_family": "engineering", "candidate_role_type": "ml",
            "job_role_type": "ml", "candidate_role_track": "applied",
            "job_role_track": "research", "candidate_seniority": "senior",
            "job_seniority": "mid", "candidate_years_experience": 6,
            "minimum_years_experience": 4, "candidate_domain": "technology",
            "job_domain": "technology", "candidate_certifications": [],
            "required_certifications": [],
        }])
        result = build_pair_features(pair).iloc[0]
        self.assertEqual(list(result.index), BASE_FEATURE_COLUMNS)
        self.assertFalse(FORBIDDEN_FEATURES & set(result.index))
        self.assertAlmostEqual(result.interview_score, .75)
        self.assertAlmostEqual(result.assessment_score, .80)
        self.assertAlmostEqual(result.required_skill_coverage, .5)
        self.assertAlmostEqual(result.preferred_skill_coverage, 1.0)
        self.assertEqual(result.role_family_match, 1.0)
        self.assertEqual(result.seniority_gap, 1.0)
        self.assertEqual(result.meets_experience_minimum, 1.0)


if __name__ == "__main__":
    unittest.main()
