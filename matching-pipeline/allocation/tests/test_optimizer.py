import pandas as pd

from allocation.optimizer import greedy_allocate, optimize_allocation


def test_global_optimizer_beats_job_by_job_greedy_choice():
    pairs = pd.DataFrame([
        {"job_id": "a", "candidate_id": "alice", "performance_probability": 0.90},
        {"job_id": "a", "candidate_id": "bob", "performance_probability": 0.88},
        {"job_id": "b", "candidate_id": "alice", "performance_probability": 0.89},
        {"job_id": "b", "candidate_id": "bob", "performance_probability": 0.50},
    ])
    greedy = greedy_allocate(pairs)
    optimal = optimize_allocation(pairs)

    assert greedy.performance_probability.sum() == 1.40
    assert optimal.performance_probability.sum() == 1.77
    assert dict(zip(optimal.job_id, optimal.candidate_id)) == {"a": "bob", "b": "alice"}


def test_ineligible_pairs_cannot_be_selected():
    pairs = pd.DataFrame([
        {"job_id": "a", "candidate_id": "alice", "performance_probability": 0.99,
         "eligible": False},
        {"job_id": "a", "candidate_id": "bob", "performance_probability": 0.60,
         "eligible": True},
    ])
    assert optimize_allocation(pairs).iloc[0].candidate_id == "bob"


def test_greedy_can_leave_a_stranded_job_unfilled_for_comparison():
    pairs = pd.DataFrame([
        {"job_id": "a", "candidate_id": "only", "performance_probability": 0.9},
        {"job_id": "b", "candidate_id": "only", "performance_probability": 0.8},
    ])
    result = greedy_allocate(pairs, require_all=False)
    assert result.job_id.tolist() == ["a"]
