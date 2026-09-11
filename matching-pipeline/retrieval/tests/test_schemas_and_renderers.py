import unittest

from retrieval.renderers import render_candidate, render_job
from retrieval.schemas import CandidateProfile, JobProfile


class SchemaAndRendererTests(unittest.TestCase):
    def test_candidate_row_uses_allowlist_and_excludes_leakage(self):
        profile = CandidateProfile.from_row(
            {
                "candidate_id": "candidate-1",
                "name": "Must Not Appear",
                "current_title": "ML Engineer",
                "skills": ["Python", "C++"],
                "fit_profile": "ideal",
                "relevance_grade": 2,
                "latent_ability": 0.99,
            }
        )
        text = render_candidate(profile)
        self.assertIn("Current title: ML Engineer", text)
        self.assertIn("Skills: C++, Python", text)
        for forbidden in ("Must Not Appear", "ideal", "0.99", "relevance_grade"):
            self.assertNotIn(forbidden, text)

    def test_rendering_is_deterministic_for_set_like_fields(self):
        first = CandidateProfile(candidate_id="c", skills=("SQL", "Python", "SQL"))
        second = CandidateProfile(candidate_id="c", skills=("Python", "SQL"))
        self.assertEqual(render_candidate(first), render_candidate(second))

    def test_job_missing_values_are_explicit(self):
        text = render_job(JobProfile(job_id="j", title="Analyst"))
        self.assertIn("Title: Analyst", text)
        self.assertIn("Minimum experience: Unknown", text)


if __name__ == "__main__":
    unittest.main()
