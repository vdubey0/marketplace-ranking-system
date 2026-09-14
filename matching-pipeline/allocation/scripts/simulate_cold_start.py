#!/usr/bin/env python3
"""Run sequential allocation rounds and measure cold-start history growth."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from allocation.outcomes import OutcomeOracle
from allocation.optimizer import greedy_allocate, optimize_allocation
from allocation.scenarios import SIMILARITY_FIELDS, has_required_certifications
from modeling.reranker import rerank_candidates
from retrieval.embeddings import embedding_matrix
from retrieval.exact_search import exact_cosine_search

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "ground-truth-generation/data/v1"
EMBEDDINGS = ROOT / "matching-pipeline/retrieval/artifacts/embeddings"
MODEL = ROOT / "matching-pipeline/modeling/artifacts/model/performance_model.joblib"
DEFAULT_OUTPUT = ROOT / "matching-pipeline/allocation/artifacts/cold_start_50_rounds"
LOG = logging.getLogger("cold_start_simulation")


def make_rounds(jobs: pd.DataFrame, count: int, size: int, cluster_size: int,
                seed: int) -> list[pd.DataFrame]:
    """Create diverse rounds composed of several internally similar job clusters."""
    if size % cluster_size:
        raise ValueError("jobs-per-round must be divisible by similar-jobs-per-cluster")
    cluster_pools = {}
    for group_number, (signature, group) in enumerate(
        jobs.groupby(SIMILARITY_FIELDS, dropna=False, sort=True)
    ):
        ordered = group.sample(frac=1, random_state=seed + group_number).reset_index(drop=True)
        cluster_pools[signature] = [
            ordered.iloc[start:start + cluster_size].copy()
            for start in range(0, len(ordered) - cluster_size + 1, cluster_size)
        ]
        cluster_pools[signature].reverse()
    clusters_per_round = size // cluster_size
    result = []
    for _ in range(count):
        active = [signature for signature, pool in cluster_pools.items() if pool]
        if len(active) < clusters_per_round:
            raise ValueError("not enough distinct job profiles to build every round")
        # Prefer profiles with more remaining jobs while allowing only one cluster
        # from a profile in any round.
        selected_signatures = sorted(
            active, key=lambda signature: (-len(cluster_pools[signature]), str(signature))
        )[:clusters_per_round]
        result.append(pd.concat(
            [cluster_pools[signature].pop() for signature in selected_signatures],
            ignore_index=True,
        ))
    return result


def history_weight(prior_assignments: pd.Series) -> pd.Series:
    """Use no history before five outcomes, then grow toward an 80% cap."""
    return ((prior_assignments - 4).clip(lower=0) * 0.10).clip(upper=0.80)


def quality(frame, score):
    if frame.empty:
        return {"rows": 0}
    y, p = frame.successful_performance.astype(int), frame[score].astype(float)
    return {"rows": len(frame), "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def full_pool_capacity(batch, candidates):
    """Maximum assignable jobs using eligibility only, with no model or labels."""
    cert_sets = [{str(x).strip().casefold() for x in values}
                 for values in candidates.certifications]
    eligible = np.array([
        [required.issubset(held) for held in cert_sets]
        for required in [{str(x).strip().casefold() for x in values}
                         for values in batch.required_certifications]
    ], dtype=bool)
    matching = maximum_bipartite_matching(csr_matrix(eligible), perm_type='column')
    return int((matching >= 0).sum()), dict(zip(batch.job_id, eligible.sum(axis=1)))


def allocate_with_expansion(job_ids, initial_k, pool_size, score, *,
                            maximum_fill=None, full_eligible_counts=None):
    """Expand retrieval without consulting outcomes; then report real vacancies."""
    k, attempts = min(initial_k, pool_size), []
    target = len(job_ids) if maximum_fill is None else maximum_fill
    while True:
        pairs = score(k)
        chosen = optimize_allocation(pairs)
        attempts.append({"retrieve_k": k, "filled": len(chosen), "maximum_fill": target})
        LOG.info("Retrieval K=%s | filled=%s/%s | achievable=%s", k, len(chosen), len(job_ids), target)
        if len(chosen) == target or k == pool_size:
            break
        k = min(pool_size, k * 2)
    vacancies = []
    eligible = pairs.loc[pairs.eligible]
    for jid in sorted(set(job_ids) - set(chosen.job_id)):
        count = (int(full_eligible_counts[jid]) if full_eligible_counts is not None
                 else int((eligible.job_id == jid).sum()))
        vacancies.append({"job_id": jid, "eligible_candidates": count,
                          "reason": "no_eligible_candidate" if count == 0 else "candidate_capacity_conflict"})
    return pairs, chosen, pd.DataFrame(vacancies, columns=["job_id", "eligible_candidates", "reason"]), attempts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--jobs-per-round", type=int, default=100)
    parser.add_argument("--similar-jobs-per-cluster", type=int, default=10)
    parser.add_argument("--retrieve-k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.rounds, args.jobs_per_round, args.similar_jobs_per_cluster, args.retrieve_k) <= 0:
        raise ValueError("Round, job, cluster and retrieval counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "round_metrics.csv").exists():
        raise ValueError("Previous run exists: choose a new --output-dir")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")])
    started = time.perf_counter()
    jobs = pd.read_parquet(DATA / "jobs.parquet")
    candidates = pd.read_parquet(DATA / "candidates.parquet")
    assessments = pd.read_parquet(DATA / "candidate_assessments.parquet")
    embeddings = pd.read_parquet(EMBEDDINGS / "candidates.parquet").sort_values("entity_id").reset_index(drop=True)
    job_vectors = pd.read_parquet(EMBEDDINGS / "jobs.parquet").set_index("entity_id")
    fitted = joblib.load(args.model)
    training = pd.read_parquet(ROOT / "matching-pipeline/modeling/artifacts/dataset/performance_pairs.parquet", columns=["job_id"])
    evaluation_jobs = jobs.loc[~jobs.job_id.isin(training.job_id)]
    rounds = make_rounds(evaluation_jobs, args.rounds, args.jobs_per_round, args.similar_jobs_per_cluster, args.seed)
    oracle = OutcomeOracle(DATA, candidates, jobs)
    matrix = embedding_matrix(embeddings)
    certs = candidates.set_index("candidate_id").certifications
    history = pd.DataFrame(0, index=pd.Index(candidates.candidate_id, name="candidate_id"),
                           columns=["prior_assignments", "prior_successes"])
    assigned_frames, vacancy_frames, metric_rows, evaluation_frames, expansion_rows = [], [], [], [], []
    for number, batch in enumerate(rounds, 1):
        LOG.info("Round %02d/%s | jobs=%s", number, args.rounds, len(batch))
        capacity, eligible_counts = full_pool_capacity(
            batch, candidates.loc[candidates.candidate_id.isin(embeddings.entity_id)])
        LOG.info("Full-pool eligibility: maximum fill=%s/%s", capacity, len(batch))
        vectors = np.stack(job_vectors.loc[batch.job_id, "embedding"])
        def score(k):
            indices, similarities = exact_cosine_search(vectors, matrix, k)
            frames = []
            for row, (_, job) in enumerate(batch.iterrows()):
                retrieved = pd.DataFrame({"candidate_id": embeddings.iloc[indices[row]].entity_id.to_numpy(),
                                          "retrieval_rank": np.arange(1, len(indices[row]) + 1),
                                          "embedding_similarity": similarities[row]})
                scored = rerank_candidates(job, retrieved, candidates, assessments, fitted, top_k=k)
                scored["job_id"] = job.job_id
                scored["eligible"] = scored.candidate_id.map(
                    lambda cid: has_required_certifications(certs.loc[cid], job.required_certifications))
                scored["prior_assignments"] = scored.candidate_id.map(history.prior_assignments)
                successes = scored.candidate_id.map(history.prior_successes)
                scored["historical_score"] = (successes + 2) / (scored.prior_assignments + 4)
                scored["history_weight"] = history_weight(scored.prior_assignments)
                scored["cold_start_probability"] = scored.performance_probability
                scored["performance_probability"] = ((1 - scored.history_weight) * scored.cold_start_probability
                                                      + scored.history_weight * scored.historical_score)
                frames.append(scored)
            return pd.concat(frames, ignore_index=True)
        pairs, optimal, vacancies, attempts = allocate_with_expansion(
            batch.job_id.tolist(), args.retrieve_k, len(embeddings), score,
            maximum_fill=capacity, full_eligible_counts=eligible_counts)
        pairs["round"] = number
        pairs.to_parquet(args.output_dir / f"scored_pairs_round_{number:03d}.parquet", index=False)
        greedy = greedy_allocate(pairs, require_all=False)
        # Reveal outcomes only after the allocation is final.
        assigned = optimal.merge(oracle.reveal(optimal), on=["job_id", "candidate_id"], validate="one_to_one")
        assigned["round"] = number
        assigned_frames.append(assigned)
        vacancies["round"] = number
        vacancy_frames.append(vacancies)
        expansion_rows.extend({"round": number, **attempt} for attempt in attempts)
        # A bounded evaluation-only sample; these outcomes never enter history.
        audit = pairs.sample(n=min(1000, len(pairs)), random_state=args.seed + number)
        audit = audit.merge(oracle.reveal(audit), on=["job_id", "candidate_id"], validate="one_to_one")
        evaluation_frames.append(audit)
        for a in assigned.itertuples(index=False):
            history.loc[a.candidate_id, "prior_assignments"] += 1
            history.loc[a.candidate_id, "prior_successes"] += int(a.successful_performance)
        metrics = {"round": number, "opportunities": len(batch), "assignments": len(assigned),
                   "maximum_fill": capacity,
                   "unfilled_jobs": len(vacancies), "fill_rate": len(assigned) / len(batch),
                   "success_rate": float(assigned.successful_performance.mean()) if len(assigned) else None,
                   "expected_successes": float(assigned.true_success_probability.sum()),
                   "greedy_unfilled_jobs": len(batch) - len(greedy),
                   "allocation_score_gain_over_greedy": float(optimal.performance_probability.sum() - greedy.performance_probability.sum()),
                   "cumulative_unique_candidates": int((history.prior_assignments > 0).sum()),
                   "candidates_with_5_plus_assignments": int((history.prior_assignments >= 5).sum()),
                   "assignments_using_history": int((assigned.history_weight > 0).sum()),
                   "final_retrieve_k": attempts[-1]["retrieve_k"],
                   "cold_start_auc": quality(audit, "cold_start_probability")["roc_auc"]}
        metric_rows.append(metrics)
        LOG.info("Round %02d | filled=%s/%s | success=%s | history_ready=%s",
                 number, len(assigned), len(batch), metrics["success_rate"], metrics["candidates_with_5_plus_assignments"])
        pd.concat(assigned_frames, ignore_index=True).to_parquet(args.output_dir / "assignments.parquet", index=False)
        pd.concat(vacancy_frames, ignore_index=True).to_csv(args.output_dir / "unfilled_jobs.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(args.output_dir / "round_metrics.csv", index=False)
        history.reset_index().to_csv(args.output_dir / "candidate_history.csv", index=False)
        pd.DataFrame(expansion_rows).to_csv(args.output_dir / "retrieval_expansions.csv", index=False)
    assignments = pd.concat(assigned_frames, ignore_index=True)
    audit = pd.concat(evaluation_frames, ignore_index=True)
    audit.to_parquet(args.output_dir / "evaluation_sample.parquet", index=False)
    total = len(assignments)
    report = {"opportunities": args.rounds * args.jobs_per_round, "assignments": total,
              "unfilled_jobs": sum(row["unfilled_jobs"] for row in metric_rows),
              "fill_rate": total / (args.rounds * args.jobs_per_round),
              "realized_success_rate": float(assignments.successful_performance.mean()) if total else None,
              "candidate_coverage": float((history.prior_assignments > 0).mean()),
              "candidates_with_5_plus_assignments": int((history.prior_assignments >= 5).sum()),
              "top_10_assignment_share": float(history.prior_assignments.nlargest(10).sum() / total) if total else None,
              "cold_start_model_at_scale": quality(audit, "cold_start_probability"),
              "blended_score_at_scale": quality(audit, "performance_probability"),
              "evaluation_scope": "Uniform sample within each final retrieval pool; default model training jobs excluded; synthetic outcomes.",
              "history_policy": "Smoothed (successes+2)/(assignments+4), 10% at 5 prior assignments increasing to 80%; heuristic, not a trained historical model.",
              "simulator_version": "independent_hash_streams_v2 extended to unmaterialized pairs",
              "runtime_seconds": time.perf_counter() - started}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    LOG.info("Finished: %s assignments from %s opportunities", total, report["opportunities"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
