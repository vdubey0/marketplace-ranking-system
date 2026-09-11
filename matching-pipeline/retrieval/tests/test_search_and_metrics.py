import unittest

import numpy as np

from retrieval.exact_search import exact_cosine_search
from retrieval.metrics import recall_at_k, recalls_at_k


class ExactSearchTests(unittest.TestCase):
    def test_cosine_search_returns_nearest_candidates(self):
        candidates = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        jobs = np.array([[1, 0]], dtype=float)
        indices, scores = exact_cosine_search(jobs, candidates, k=2)
        np.testing.assert_array_equal(indices, [[0, 2]])
        self.assertAlmostEqual(float(scores[0, 0]), 1.0)

    def test_ties_are_ordered_by_candidate_row(self):
        indices, _ = exact_cosine_search(
            np.array([[1, 0]]), np.array([[1, 1], [1, -1]]), k=2
        )
        np.testing.assert_array_equal(indices, [[0, 1]])

    def test_rejects_zero_vectors(self):
        with self.assertRaisesRegex(ValueError, "zero vector"):
            exact_cosine_search(np.array([[1, 0]]), np.array([[0, 0]]), k=1)


class RecallTests(unittest.TestCase):
    def test_recall_at_multiple_cutoffs(self):
        result = recalls_at_k(["a", "x", "b"], {"a", "b", "c"}, [1, 3])
        self.assertEqual(result, {1: 1 / 3, 3: 2 / 3})

    def test_duplicates_do_not_get_extra_credit(self):
        self.assertEqual(recall_at_k(["a", "a"], {"a", "b"}, 2), 0.5)

    def test_zero_positive_job_is_not_scored_as_zero(self):
        self.assertIsNone(recall_at_k(["a"], set(), 1))


if __name__ == "__main__":
    unittest.main()
