"""Turn exact-search results into auditable per-job and aggregate metrics."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def evaluate_retrieval(
    retrieval: pd.DataFrame,
    relevance: pd.DataFrame,
    ks: Iterable[int],
    *,
    minimum_grade: int = 1,
) -> tuple[pd.DataFrame, dict]:
    required_results = {"job_id", "candidate_id", "rank"}
    required_labels = {"job_id", "candidate_id", "relevance_grade"}
    if not required_results.issubset(retrieval.columns):
        raise ValueError(f"retrieval requires columns {sorted(required_results)}")
    if not required_labels.issubset(relevance.columns):
        raise ValueError(f"relevance requires columns {sorted(required_labels)}")
    cutoffs = sorted(set(int(k) for k in ks))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("ks must contain positive integers")

    relevant = relevance.loc[relevance.relevance_grade >= minimum_grade,
                              ["job_id", "candidate_id"]].drop_duplicates()
    relevant_sets = relevant.groupby("job_id").candidate_id.agg(set).to_dict()
    retrieved = retrieval.sort_values(["job_id", "rank"]).groupby("job_id")
    rows = []
    for job_id, group in retrieved:
        positives = relevant_sets.get(job_id, set())
        row = {"job_id": job_id, "relevant_count": len(positives)}
        for k in cutoffs:
            found = set(group.loc[group["rank"] <= k, "candidate_id"])
            hits = len(found & positives)
            row[f"hits_at_{k}"] = hits
            row[f"recall_at_{k}"] = hits / len(positives) if positives else None
        rows.append(row)
    per_job = pd.DataFrame(rows)
    evaluable = per_job.loc[per_job.relevant_count > 0]
    summary = {
        "minimum_relevance_grade": minimum_grade,
        "jobs_returned": len(per_job),
        "evaluable_jobs": len(evaluable),
        "zero_positive_jobs": int((per_job.relevant_count == 0).sum()),
        "cutoffs": {},
    }
    for k in cutoffs:
        values = evaluable[f"recall_at_{k}"]
        summary["cutoffs"][str(k)] = {
            "macro_recall": float(values.mean()) if len(values) else None,
            "micro_recall": (
                float(evaluable[f"hits_at_{k}"].sum() / evaluable.relevant_count.sum())
                if evaluable.relevant_count.sum() else None
            ),
            "median_recall": float(values.median()) if len(values) else None,
            "p10_recall": float(values.quantile(0.10)) if len(values) else None,
            "worst_job_recall": float(values.min()) if len(values) else None,
        }
    return per_job, summary
