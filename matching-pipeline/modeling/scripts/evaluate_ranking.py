#!/usr/bin/env python3
"""Compare LightGBM candidate ordering with cosine-similarity ordering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modeling.ranking_metrics import evaluate_rankings

MODELING_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=MODELING_ROOT / "artifacts/dataset/performance_pairs.parquet")
    parser.add_argument("--predictions", type=Path,
                        default=MODELING_ROOT / "artifacts/model/out_of_fold_predictions.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=MODELING_ROOT / "artifacts/ranking_evaluation")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 25])
    args = parser.parse_args()

    data = pd.read_parquet(
        args.dataset, columns=["job_id", "candidate_id", "embedding_similarity"]
    )
    predictions = pd.read_parquet(args.predictions)
    frame = predictions.merge(
        data, on=["job_id", "candidate_id"], how="left", validate="one_to_one"
    )
    if frame.embedding_similarity.isna().any():
        raise ValueError("Some validation predictions have no matching cosine similarity")
    per_job = evaluate_rankings(
        frame, ["predicted_probability", "embedding_similarity"], args.ks
    )
    summary = {"jobs": len(per_job), "cutoffs": {}}
    for k in sorted(set(args.ks)):
        model_success = per_job[f"predicted_probability_success_at_{k}"].mean()
        cosine_success = per_job[f"embedding_similarity_success_at_{k}"].mean()
        model_ndcg = per_job[f"predicted_probability_ndcg_at_{k}"].mean()
        cosine_ndcg = per_job[f"embedding_similarity_ndcg_at_{k}"].mean()
        summary["cutoffs"][str(k)] = {
            "model_success_rate": float(model_success),
            "cosine_success_rate": float(cosine_success),
            "success_rate_lift": float(model_success - cosine_success),
            "model_ndcg": float(model_ndcg),
            "cosine_ndcg": float(cosine_ndcg),
            "ndcg_lift": float(model_ndcg - cosine_ndcg),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_job.to_parquet(args.output_dir / "per_job_metrics.parquet", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

