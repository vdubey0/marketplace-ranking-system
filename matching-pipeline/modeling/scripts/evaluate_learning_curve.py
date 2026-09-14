#!/usr/bin/env python3
"""Measure whether more labeled jobs still improve the LightGBM model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from modeling.features import BASE_FEATURE_COLUMNS
from modeling.ranking_metrics import ndcg_at_k

MODELING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = MODELING_ROOT / "artifacts/dataset/performance_pairs.parquet"
DEFAULT_OUTPUT = MODELING_ROOT / "artifacts/learning_curve"
FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", num_leaves=31, min_child_samples=100,
        learning_rate=0.05, n_estimators=400, reg_alpha=0.0,
        reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def job_ranking_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    scored = frame[["job_id", "candidate_id", "successful_performance"]].copy()
    scored["probability"] = probabilities
    top5_rates, ndcgs = [], []
    for _, group in scored.groupby("job_id"):
        ranked = group.sort_values(
            ["probability", "candidate_id"], ascending=[False, True]
        )
        top5_rates.append(float(ranked.head(5).successful_performance.mean()))
        ndcg = ndcg_at_k(
            group.successful_performance.to_numpy(), group.probability.to_numpy(), 5
        )
        if ndcg is not None:
            ndcgs.append(ndcg)
    return {
        "top5_success_rate": float(np.mean(top5_rates)),
        "ndcg_at_5": float(np.mean(ndcgs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    data = pd.read_parquet(args.dataset)
    jobs = np.array(sorted(data.job_id.unique()))
    split_rng = np.random.default_rng(args.seed)
    split_rng.shuffle(jobs)
    holdout_count = round(0.2 * len(jobs))
    holdout_jobs = set(jobs[:holdout_count])
    development_jobs = jobs[holdout_count:]
    holdout = data.loc[data.job_id.isin(holdout_jobs)].reset_index(drop=True)

    rows = []
    for repeat in range(args.repeats):
        ordered_jobs = development_jobs.copy()
        np.random.default_rng(args.seed + repeat).shuffle(ordered_jobs)
        for fraction in FRACTIONS:
            job_count = max(1, round(fraction * len(ordered_jobs)))
            training_jobs = set(ordered_jobs[:job_count])
            training = data.loc[data.job_id.isin(training_jobs)]
            fitted = make_model(args.seed + repeat)
            fitted.fit(training[BASE_FEATURE_COLUMNS], training.successful_performance)
            probabilities = fitted.predict_proba(holdout[BASE_FEATURE_COLUMNS])[:, 1]
            rows.append({
                "training_fraction": fraction,
                "repeat": repeat + 1,
                "training_jobs": job_count,
                "training_rows": len(training),
                "roc_auc": float(roc_auc_score(holdout.successful_performance, probabilities)),
                "average_precision": float(average_precision_score(
                    holdout.successful_performance, probabilities
                )),
                **job_ranking_metrics(holdout, probabilities),
            })

    raw = pd.DataFrame(rows)
    summary = (raw.groupby("training_fraction", as_index=False)
               .agg(training_jobs=("training_jobs", "max"),
                    mean_training_rows=("training_rows", "mean"),
                    roc_auc=("roc_auc", "mean"),
                    roc_auc_std=("roc_auc", "std"),
                    average_precision=("average_precision", "mean"),
                    top5_success_rate=("top5_success_rate", "mean"),
                    top5_success_rate_std=("top5_success_rate", "std"),
                    ndcg_at_5=("ndcg_at_5", "mean")))
    final_gain = float(summary.iloc[-1].top5_success_rate - summary.iloc[-2].top5_success_rate)
    conclusion = "widen_dataset" if final_gain >= 0.005 else "combine_retrieval_and_modeling"
    report = {
        "holdout_jobs": holdout_count,
        "development_jobs": len(development_jobs),
        "repeats": args.repeats,
        "decision_rule": "widen only if 100% data improves holdout top-5 by at least 0.5 percentage points over 50%",
        "top5_gain_from_50_to_100": final_gain,
        "conclusion": conclusion,
        "results": summary.to_dict(orient="records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "runs.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
