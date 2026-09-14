"""Validate completed relevance and performance labels and save a quality report."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "ground-truth-generation/data/v1"


def validate(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    candidates = pd.read_parquet(data_dir / "candidates.parquet", columns=["candidate_id", "name"])
    candidate_hidden = pd.read_parquet(data_dir / "candidate_hidden.parquet", columns=["candidate_id"])
    jobs = pd.read_parquet(data_dir / "jobs.parquet", columns=["job_id"])
    job_hidden = pd.read_parquet(data_dir / "job_hidden.parquet", columns=["job_id"])

    assert len(candidates) == manifest["candidate_count"]
    assert candidates.candidate_id.is_unique and candidates.name.is_unique
    assert candidates.name.notna().all() and candidates.name.str.strip().ne("").all()
    assert set(candidates.candidate_id) == set(candidate_hidden.candidate_id)
    assert len(jobs) == manifest["job_count"] and jobs.job_id.is_unique
    assert set(jobs.job_id) == set(job_hidden.job_id)

    negatives = pd.read_parquet(data_dir / "relevance_negative_sample.parquet")
    negative_counts = negatives.groupby("job_id").size()
    assert len(negative_counts) == len(jobs)
    assert negative_counts.nunique() == 1
    assert negatives.relevance_grade.eq(0).all() and negatives.relevant.eq(False).all()

    audit = pd.read_csv(data_dir / "relevance_audit_sample.csv")
    audit_counts = audit.groupby("relevance_grade").size().to_dict()
    assert set(audit_counts) == {0, 1, 2}

    relevance_counts = Counter()
    for batch in pq.ParquetFile(data_dir / "relevance_ground_truth.parquet").iter_batches(
        columns=["relevance_grade", "relevant", "relevance_score"], batch_size=250_000
    ):
        frame = batch.to_pandas()
        assert frame.relevance_grade.isin([1, 2]).all() and frame.relevant.eq(True).all()
        assert frame.relevance_score.between(0, 1).all()
        relevance_counts.update(frame.relevance_grade.astype(int))

    outcome_counts = Counter()
    probability_sums = defaultdict(float)
    success_counts = Counter()
    probability_min, probability_max = 1.0, 0.0
    outcome_rows = 0
    calibration = defaultdict(lambda: {"count": 0, "probability_sum": 0.0, "successes": 0})
    for batch in pq.ParquetFile(data_dir / "marketplace_outcomes.parquet").iter_batches(
        columns=["relevance_grade", "true_success_probability", "potential_success",
                 "successful_performance", "selected", "observed_success"],
        batch_size=250_000,
    ):
        frame = batch.to_pandas()
        assert frame.true_success_probability.between(0, 1).all()
        assert frame.potential_success.equals(frame.successful_performance)
        assert frame.selected.isna().all() and frame.observed_success.isna().all()
        probability_min = min(probability_min, float(frame.true_success_probability.min()))
        probability_max = max(probability_max, float(frame.true_success_probability.max()))
        outcome_rows += len(frame)
        bins = (frame.true_success_probability.mul(10).astype(int).clip(upper=9))
        for bin_number, group in frame.groupby(bins):
            bucket = calibration[int(bin_number)]
            bucket["count"] += len(group)
            bucket["probability_sum"] += float(group.true_success_probability.sum())
            bucket["successes"] += int(group.successful_performance.sum())
        for grade, group in frame.groupby("relevance_grade"):
            grade = int(grade)
            outcome_counts[grade] += len(group)
            probability_sums[grade] += float(group.true_success_probability.sum())
            success_counts[grade] += int(group.successful_performance.sum())

    assert outcome_rows == manifest["outcome_pairs"]
    mean_probability = {str(g): probability_sums[g] / outcome_counts[g] for g in sorted(outcome_counts)}
    success_rate = {str(g): success_counts[g] / outcome_counts[g] for g in sorted(outcome_counts)}
    assert list(mean_probability.values()) == sorted(mean_probability.values())
    calibration_rows = []
    for bin_number, values in sorted(calibration.items()):
        mean_p = values["probability_sum"] / values["count"]
        rate = values["successes"] / values["count"]
        calibration_rows.append({"probability_bin": f"{bin_number / 10:.1f}-{(bin_number + 1) / 10:.1f}",
                                 "count": values["count"], "mean_probability": mean_p,
                                 "realized_success_rate": rate, "absolute_error": abs(rate - mean_p)})
    well_populated = [row for row in calibration_rows if row["count"] >= 10_000]
    assert max(row["absolute_error"] for row in well_populated) < 0.02

    report = {
        "status": "passed",
        "candidate_count": len(candidates),
        "unique_candidate_names": int(candidates.name.nunique()),
        "job_count": len(jobs),
        "relevance_grade_counts": {
            "0": int(len(negatives)),
            "1": int(relevance_counts[1]),
            "2": int(relevance_counts[2]),
        },
        "negative_examples_per_job": int(negative_counts.iloc[0]),
        "audit_grade_counts": {str(k): int(v) for k, v in audit_counts.items()},
        "performance_label_count": outcome_rows,
        "true_success_probability_range": [probability_min, probability_max],
        "mean_success_probability_by_relevance_grade": mean_probability,
        "realized_success_rate_by_relevance_grade": success_rate,
        "calibration_by_probability_bin": calibration_rows,
        "selected_and_observed_labels_are_deferred": True,
    }
    (data_dir / "labeling_quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(validate(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
