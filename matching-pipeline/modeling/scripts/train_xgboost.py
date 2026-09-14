#!/usr/bin/env python3
"""Train XGBoost on the same job-grouped folds used for LightGBM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from modeling.features import BASE_FEATURE_COLUMNS, FORBIDDEN_FEATURES

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "artifacts/dataset/performance_pairs.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/model_xgboost"


def metrics(target, probability) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def new_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    data = pd.read_parquet(args.dataset)
    if FORBIDDEN_FEATURES & set(BASE_FEATURE_COLUMNS):
        raise AssertionError("Forbidden feature configured")
    X = data[BASE_FEATURE_COLUMNS]
    target = data.successful_performance.astype(int)
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    out_of_fold = np.zeros(len(data), dtype=np.float32)
    fold_rows = []
    importances = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(X, target, data.job_id), start=1
    ):
        model = new_model(args.seed + fold)
        model.fit(X.iloc[train_index], target.iloc[train_index])
        probability = model.predict_proba(X.iloc[validation_index])[:, 1]
        out_of_fold[validation_index] = probability
        fold_rows.append({
            "fold": fold,
            "train_rows": len(train_index),
            "validation_rows": len(validation_index),
            "validation_jobs": int(data.job_id.iloc[validation_index].nunique()),
            **metrics(target.iloc[validation_index], probability),
        })
        importances.append(model.feature_importances_)

    prevalence = float(target.mean())
    baseline = metrics(target, np.full(len(target), prevalence))
    overall = metrics(target, out_of_fold)
    report = {
        "model": "XGBClassifier",
        "folds": fold_rows,
        "overall_oof": overall,
        "constant_probability_baseline": {"probability": prevalence, **baseline},
        "rows": len(data),
        "jobs": int(data.job_id.nunique()),
        "feature_count": len(BASE_FEATURE_COLUMNS),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "job_id": data.job_id,
        "candidate_id": data.candidate_id,
        "target": target,
        "predicted_probability": out_of_fold,
    }).to_parquet(args.output_dir / "out_of_fold_predictions.parquet", index=False)
    pd.DataFrame({
        "feature": BASE_FEATURE_COLUMNS,
        "mean_gain_importance": np.mean(importances, axis=0),
    }).sort_values("mean_gain_importance", ascending=False).to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    final_model = new_model(args.seed)
    final_model.fit(X, target)
    joblib.dump(final_model, args.output_dir / "performance_model.joblib")
    (args.output_dir / "cross_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
