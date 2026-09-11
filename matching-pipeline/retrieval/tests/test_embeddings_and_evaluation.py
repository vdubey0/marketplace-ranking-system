import tempfile
import unittest
from pathlib import Path

import pandas as pd

from retrieval.embeddings import embed_records
from retrieval.evaluation import evaluate_retrieval


class EmbeddingArtifactTests(unittest.TestCase):
    def test_unchanged_records_are_reused(self):
        calls = []
        def embed(texts):
            calls.append(texts)
            return [[float(len(text)), 1.0] for text in texts]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.parquet"
            embed_records([("a", "alpha"), ("b", "beta")], path, embed,
                          model="fake", dimensions=2, batch_size=1)
            embed_records([("a", "alpha"), ("b", "beta")], path, embed,
                          model="fake", dimensions=2, batch_size=1)
        self.assertEqual(calls, [["alpha"], ["beta"]])


class EvaluationTests(unittest.TestCase):
    def test_macro_micro_and_zero_positive_jobs(self):
        retrieval = pd.DataFrame([
            {"job_id": "j1", "candidate_id": "a", "rank": 1},
            {"job_id": "j1", "candidate_id": "x", "rank": 2},
            {"job_id": "j2", "candidate_id": "x", "rank": 1},
        ])
        labels = pd.DataFrame([
            {"job_id": "j1", "candidate_id": "a", "relevance_grade": 2},
            {"job_id": "j1", "candidate_id": "b", "relevance_grade": 1},
        ])
        per_job, summary = evaluate_retrieval(retrieval, labels, [1, 2])
        self.assertEqual(summary["evaluable_jobs"], 1)
        self.assertEqual(summary["zero_positive_jobs"], 1)
        self.assertEqual(summary["cutoffs"]["1"]["micro_recall"], 0.5)
        self.assertTrue(pd.isna(per_job.loc[per_job.job_id == "j2", "recall_at_1"]).all())


if __name__ == "__main__":
    unittest.main()
