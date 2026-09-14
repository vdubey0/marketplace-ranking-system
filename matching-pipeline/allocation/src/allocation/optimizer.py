"""Binary linear optimization for one-candidate-per-job allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

REQUIRED_COLUMNS = {"job_id", "candidate_id", "performance_probability"}


def _feasible_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs are missing columns: {sorted(missing)}")
    eligible = (pairs["eligible"].astype(bool)
                if "eligible" in pairs.columns else pd.Series(True, index=pairs.index))
    feasible = pairs.loc[eligible].copy()
    if feasible.duplicated(["job_id", "candidate_id"]).any():
        raise ValueError("each candidate-job pair must be unique")
    if not feasible.performance_probability.between(0, 1).all():
        raise ValueError("performance probabilities must be between zero and one")
    return feasible.reset_index(drop=True)


def optimize_allocation(pairs: pd.DataFrame) -> pd.DataFrame:
    """Maximize filled jobs first, then success probability at that fill count.

    An omitted job is unfilled (its implicit vacancy variable is one).
    Callers retain the input job roster to report vacancies, including empty pools.
    """
    feasible = _feasible_pairs(pairs)
    if feasible.empty:
        return feasible
    jobs = sorted(feasible.job_id.unique())
    candidates = sorted(feasible.candidate_id.unique())
    job_row = {job_id: row for row, job_id in enumerate(jobs)}
    candidate_row = {candidate_id: len(jobs) + row
                     for row, candidate_id in enumerate(candidates)}
    matrix = lil_matrix((len(jobs) + len(candidates), len(feasible)), dtype=float)
    for column, pair in enumerate(feasible.itertuples(index=False)):
        matrix[job_row[pair.job_id], column] = 1.0
        matrix[candidate_row[pair.candidate_id], column] = 1.0
    lower = np.zeros(len(jobs) + len(candidates))
    upper = np.ones(len(jobs) + len(candidates))
    constraints = LinearConstraint(matrix.tocsr(), lower, upper)
    result = milp(
        c=-np.ones(len(feasible)),
        integrality=np.ones(len(feasible)),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options={"presolve": True, "mip_rel_gap": 0},
    )
    if not result.success:
        raise RuntimeError(f"allocation failed: {result.message}")
    fill = int(np.rint(result.x.sum()))
    result = milp(
        c=-feasible.performance_probability.to_numpy(),
        integrality=np.ones(len(feasible)), bounds=Bounds(0, 1),
        constraints=[constraints, LinearConstraint(np.ones((1, len(feasible))), fill, fill)],
        options={"presolve": True, "mip_rel_gap": 0},
    )
    if not result.success:
        raise RuntimeError(f"allocation score optimization failed: {result.message}")
    chosen = feasible.loc[result.x > 0.5].copy()
    return chosen.sort_values("job_id").reset_index(drop=True)


def greedy_allocate(pairs: pd.DataFrame, *, require_all: bool = True) -> pd.DataFrame:
    """Assign each job's best remaining candidate in sorted job order.

    With ``require_all=False``, a job stranded by an earlier greedy choice is
    recorded as unfilled instead of stopping a comparison with global allocation.
    """
    feasible = _feasible_pairs(pairs)
    chosen_rows, used = [], set()
    for job_id in sorted(pairs.job_id.unique()):
        choices = feasible.loc[
            (feasible.job_id == job_id) & ~feasible.candidate_id.isin(used)
        ].sort_values(
            ["performance_probability", "candidate_id"], ascending=[False, True]
        )
        if choices.empty:
            if require_all:
                raise RuntimeError(f"greedy allocation cannot fill job {job_id}")
            continue
        choice = choices.iloc[0]
        chosen_rows.append(choice)
        used.add(choice.candidate_id)
    return pd.DataFrame(chosen_rows, columns=feasible.columns).reset_index(drop=True)
