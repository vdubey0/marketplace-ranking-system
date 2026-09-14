import unittest

import pandas as pd

from modeling.ranking_metrics import evaluate_rankings, ndcg_at_k


class RankingMetricTests(unittest.TestCase):
    def test_perfect_ranking_has_ndcg_one(self):
        self.assertEqual(ndcg_at_k([1, 0, 1], [0.9, 0.1, 0.8], 3), 1.0)

    def test_job_without_success_has_undefined_ndcg(self):
        self.assertIsNone(ndcg_at_k([0, 0], [0.9, 0.1], 2))

    def test_evaluates_each_job_independently(self):
        frame = pd.DataFrame([
            {"job_id": "a", "candidate_id": "1", "target": 1, "model": .9},
            {"job_id": "a", "candidate_id": "2", "target": 0, "model": .1},
            {"job_id": "b", "candidate_id": "1", "target": 0, "model": .9},
            {"job_id": "b", "candidate_id": "2", "target": 1, "model": .1},
        ])
        result = evaluate_rankings(frame, ["model"], [1])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.model_success_at_1.tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
