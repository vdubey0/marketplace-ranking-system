"""Per-job metrics for comparing candidate ranking strategies."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def ndcg_at_k(targets: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    """Binary NDCG, or None when a job has no successful candidates."""
    if k <= 0:
        raise ValueError("k must be positive")
    targets = np.asarray(targets, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.shape != scores.shape:
        raise ValueError("targets and scores must have the same shape")
    if targets.sum() == 0:
        return None
    limit = min(k, len(targets))
    discounts = 1.0 / np.log2(np.arange(2, limit + 2))
    order = np.argsort(-scores, kind="stable")[:limit]
    dcg = float(np.sum(targets[order] * discounts))
    ideal = np.sort(targets)[::-1][:limit]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg


def evaluate_rankings(
    frame: pd.DataFrame,
    score_columns: Iterable[str],
    ks: Iterable[int],
) -> pd.DataFrame:
    """Return one auditable row per job, scoring every requested strategy."""
    cutoffs = sorted(set(int(k) for k in ks))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("ks must contain positive integers")
    strategies = list(score_columns)
    required = {"job_id", "candidate_id", "target", *strategies}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    rows = []
    for job_id, group in frame.groupby("job_id", sort=True):
        row = {"job_id": job_id, "candidate_count": len(group),
               "overall_success_rate": float(group.target.mean())}
        for strategy in strategies:
            ranked = group.sort_values([strategy, "candidate_id"], ascending=[False, True])
            for k in cutoffs:
                top = ranked.head(k)
                row[f"{strategy}_success_at_{k}"] = float(top.target.mean())
                row[f"{strategy}_ndcg_at_{k}"] = ndcg_at_k(
                    group.target.to_numpy(), group[strategy].to_numpy(), k
                )
        rows.append(row)
    return pd.DataFrame(rows)

