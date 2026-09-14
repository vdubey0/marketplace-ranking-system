#!/usr/bin/env python3
"""Tune LightGBM for top-5 candidate quality with an untouched job holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from modeling.features import BASE_FEATURE_COLUMNS
from modeling.ranking_metrics import ndcg_at_k

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "artifacts/dataset/performance_pairs.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/model_tuned"

CONFIGS = [
    {"name": "default", "num_leaves": 31, "min_child_samples": 100,
     "learning_rate": .05, "n_estimators": 400, "reg_alpha": 0., "reg_lambda": 1.},
    {"name": "small_leaves", "num_leaves": 15, "min_child_samples": 100,
     "learning_rate": .05, "n_estimators": 400, "reg_alpha": 0., "reg_lambda": 1.},
    {"name": "large_leaves", "num_leaves": 63, "min_child_samples": 100,
     "learning_rate": .05, "n_estimators": 400, "reg_alpha": 0., "reg_lambda": 1.},
    {"name": "larger_leaves", "num_leaves": 127, "min_child_samples": 100,
     "learning_rate": .05, "n_estimators": 400, "reg_alpha": 0., "reg_lambda": 1.},
    {"name": "more_regularized", "num_leaves": 31, "min_child_samples": 250,
     "learning_rate": .05, "n_estimators": 400, "reg_alpha": .5, "reg_lambda": 5.},
    {"name": "less_regularized", "num_leaves": 31, "min_child_samples": 50,
     "learning_rate": .05, "n_estimators": 400, "reg_alpha": 0., "reg_lambda": .1},
    {"name": "slow_learning", "num_leaves": 31, "min_child_samples": 100,
     "learning_rate": .03, "n_estimators": 650, "reg_alpha": 0., "reg_lambda": 1.},
    {"name": "fast_learning", "num_leaves": 31, "min_child_samples": 100,
     "learning_rate": .08, "n_estimators": 250, "reg_alpha": 0., "reg_lambda": 1.},
]


def model(config: dict, seed: int) -> lgb.LGBMClassifier:
    parameters = {key: value for key, value in config.items() if key != "name"}
    return lgb.LGBMClassifier(
        objective="binary", subsample=.8, colsample_bytree=.8, random_state=seed,
        n_jobs=-1, verbosity=-1, **parameters,
    )


def ranking_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    scored = frame[["job_id", "candidate_id", "successful_performance"]].copy()
    scored["probability"] = probabilities
    success_rates, ndcgs = [], []
    for _, group in scored.groupby("job_id"):
        ranked = group.sort_values(["probability", "candidate_id"], ascending=[False, True])
        success_rates.append(float(ranked.head(5).successful_performance.mean()))
        ndcgs.append(ndcg_at_k(group.successful_performance.to_numpy(),
                               group.probability.to_numpy(), 5))
    defined_ndcgs = [value for value in ndcgs if value is not None]
    return {"top5_success_rate": float(np.mean(success_rates)),
            "ndcg_at_5": float(np.mean(defined_ndcgs)) if defined_ndcgs else None}


def classification_metrics(target, probabilities) -> dict[str, float]:
    return {"roc_auc": float(roc_auc_score(target, probabilities)),
            "average_precision": float(average_precision_score(target, probabilities)),
            "log_loss": float(log_loss(target, probabilities)),
            "brier_score": float(brier_score_loss(target, probabilities))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    data = pd.read_parquet(args.dataset)
    jobs = np.array(sorted(data.job_id.unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(jobs)
    holdout_jobs = set(jobs[:round(.2 * len(jobs))])
    development = data.loc[~data.job_id.isin(holdout_jobs)].reset_index(drop=True)
    holdout = data.loc[data.job_id.isin(holdout_jobs)].reset_index(drop=True)

    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
    tuning_rows = []
    for config in CONFIGS:
        fold_rows = []
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(development, groups=development.job_id), start=1
        ):
            fitted = model(config, args.seed + fold)
            fitted.fit(development.loc[train_index, BASE_FEATURE_COLUMNS],
                       development.loc[train_index, "successful_performance"])
            probabilities = fitted.predict_proba(
                development.loc[validation_index, BASE_FEATURE_COLUMNS]
            )[:, 1]
            fold_rows.append(ranking_metrics(development.loc[validation_index], probabilities))
        tuning_rows.append({"name": config["name"],
                            "mean_top5_success_rate": float(np.mean([x["top5_success_rate"] for x in fold_rows])),
                            "mean_ndcg_at_5": float(np.mean([x["ndcg_at_5"] for x in fold_rows])),
                            "folds": fold_rows, "parameters": config})
    tuning_rows.sort(key=lambda row: (row["mean_top5_success_rate"], row["mean_ndcg_at_5"]), reverse=True)
    winner = tuning_rows[0]

    holdout_results = {}
    for config in (CONFIGS[0], winner["parameters"]):
        fitted = model(config, args.seed)
        fitted.fit(development[BASE_FEATURE_COLUMNS], development.successful_performance)
        probabilities = fitted.predict_proba(holdout[BASE_FEATURE_COLUMNS])[:, 1]
        holdout_results[config["name"]] = {
            **ranking_metrics(holdout, probabilities),
            **classification_metrics(holdout.successful_performance, probabilities),
        }
    deployed_name = (winner["name"] if holdout_results[winner["name"]]["top5_success_rate"]
                     > holdout_results["default"]["top5_success_rate"] else "default")
    deployed_config = next(config for config in CONFIGS if config["name"] == deployed_name)
    final_model = model(deployed_config, args.seed)
    final_model.fit(data[BASE_FEATURE_COLUMNS], data.successful_performance)

    report = {"selection_metric": "mean job-level top5 success rate",
              "development_jobs": int(development.job_id.nunique()),
              "untouched_holdout_jobs": int(holdout.job_id.nunique()),
              "tuning_results": tuning_rows, "winner": winner["name"],
              "holdout_results": holdout_results,
              "deployed_configuration": deployed_name,
              "deployment_reason": ("tuned winner improved holdout top-5 success"
                                    if deployed_name != "default" else
                                    "tuned winner did not improve holdout top-5 success")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.output_dir / "performance_model.joblib")
    (args.output_dir / "tuning_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
