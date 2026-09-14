#!/usr/bin/env python3
"""Train LightGBM with candidate-job leakage prevented by job-grouped folds."""
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

from modeling.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, FORBIDDEN_FEATURES

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "artifacts/dataset/performance_pairs.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/model"


def metrics(y, probability) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def new_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--feature-set", choices=("base", "pca"), default="base")
    args = parser.parse_args()
    data = pd.read_parquet(args.dataset)
    feature_columns = FEATURE_COLUMNS if args.feature_set == "pca" else BASE_FEATURE_COLUMNS
    if FORBIDDEN_FEATURES & set(feature_columns):
        raise AssertionError("Forbidden feature configured")
    X = data[feature_columns]
    y = data.successful_performance.astype(int)
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(data), dtype=np.float32)
    fold_rows = []
    importances = []
    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y, data.job_id), start=1):
        model = new_model(args.seed + fold)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        probability = model.predict_proba(X.iloc[validation_index])[:, 1]
        oof[validation_index] = probability
        fold_rows.append({"fold": fold, "train_rows": len(train_index),
                          "validation_rows": len(validation_index),
                          "validation_jobs": int(data.job_id.iloc[validation_index].nunique()),
                          **metrics(y.iloc[validation_index], probability)})
        importances.append(model.feature_importances_)
    prevalence = float(y.mean())
    baseline = metrics(y, np.full(len(y), prevalence))
    overall = metrics(y, oof)
    report = {
        "folds": fold_rows,
        "overall_oof": overall,
        "constant_probability_baseline": {"probability": prevalence, **baseline},
        "improvement_over_baseline": {
            "roc_auc": overall["roc_auc"] - baseline["roc_auc"],
            "average_precision": overall["average_precision"] - baseline["average_precision"],
            "log_loss_reduction": baseline["log_loss"] - overall["log_loss"],
            "brier_score_reduction": baseline["brier_score"] - overall["brier_score"],
        },
        "feature_set": args.feature_set, "feature_count": len(feature_columns),
        "rows": len(data), "jobs": int(data.job_id.nunique()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"job_id": data.job_id, "candidate_id": data.candidate_id,
                  "target": y, "predicted_probability": oof}).to_parquet(
        args.output_dir / "out_of_fold_predictions.parquet", index=False
    )
    pd.DataFrame({"feature": feature_columns,
                  "mean_split_importance": np.mean(importances, axis=0)}).sort_values(
        "mean_split_importance", ascending=False
    ).to_csv(args.output_dir / "feature_importance.csv", index=False)
    final_model = new_model(args.seed)
    final_model.fit(X, y)
    joblib.dump(final_model, args.output_dir / "performance_model.joblib")
    (args.output_dir / "cross_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
