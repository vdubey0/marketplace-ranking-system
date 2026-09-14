# Allocation

This stage assigns candidates across a batch of similar jobs using binary linear
optimization. It first maximizes filled jobs, then maximizes total predicted
success at that fill count. Each candidate and job can have at most one assignment.
Only retrieved, hard-eligible candidate-job pairs are decision variables.

Run the demonstration:

```bash
PYTHONPATH=matching-pipeline/retrieval/src:matching-pipeline/modeling/src:matching-pipeline/allocation/src \
  .venv/bin/python matching-pipeline/allocation/scripts/run_allocation.py
```

The generated artifacts include the similar jobs, all scored feasible pairs,
greedy assignments, globally optimal assignments, and a comparison summary.

Run 50 sequential cold-start rounds (100 jobs per round):

```bash
PYTHONPATH=matching-pipeline/retrieval/src:matching-pipeline/modeling/src:matching-pipeline/allocation/src \
  .venv/bin/python matching-pipeline/allocation/scripts/simulate_cold_start.py
```

Each 100-job round contains ten clusters of ten similar jobs. This preserves
candidate competition without making all 100 openings nearly identical. The
simulation retrieves 500 candidates per job for assignment flexibility. Progress is printed to the terminal and saved in
`artifacts/cold_start_50_rounds/run.log`. Candidate history receives no weight
until five prior assignments; the summary includes model, allocation, coverage,
concentration, and history-growth metrics.

Certification jobs are included. Missing outcome rows never affect eligibility.
After assignment, the existing simulator generates the pair's outcome on demand;
its original seed and independently salted streams are preserved. Generated
outcomes for a bounded evaluation sample are never added to candidate history.

An eligibility-only matching check first computes the maximum possible fill across
the entire candidate pool. Retrieval expands only until that count is reached
(or the entire pool is retrieved), avoiding repeated scoring for impossible vacancies.
Remaining vacancies are reported as no eligible candidate or a candidate-capacity
conflict. Thus 50 rounds produce 5,000 opportunities, potentially fewer assignments.
The script does not add rounds silently to reach a target number of assignments.

Logs and completed-round checkpoints include fill rate, vacancies, retrieval
expansion, success rate, model AUC/calibration, coverage, and history growth.
The default model's training jobs are excluded. History blending remains an
experimental heuristic; this run does not establish a causal benefit from history.
Choose a new `--output-dir` for each completed experiment.
