# Ground-truth generation

This local stage creates relevance and performance labels. It makes no API calls.

```bash
.venv/bin/python ground-truth-generation/scripts/generate_ground_truth.py
```

Outputs go to `data/v1/`: public candidates with names, normalized jobs, relevance
truth, a review CSV, hidden candidate/job traits, and simulated outcomes. Names,
`fit_profile`, `is_synthetic`, latent traits, and true probabilities must never be
used as ranking features.

`relevance_grade` is the retrieval label: `0` means irrelevant, `1` means
borderline, and `2` means clearly relevant. `successful_performance` is the binary
ranking label. It is a readable alias of the simulator's `potential_success`
column. `selected` and `observed_success` remain empty until a later stage models
historical marketplace selection.

Validate the completed label files and recreate `labeling_quality_report.json`:

```bash
.venv/bin/python ground-truth-generation/scripts/validate_ground_truth.py
```

Regenerate only performance outcomes after changing the outcome simulator:

```bash
.venv/bin/python ground-truth-generation/scripts/regenerate_outcomes.py
```

Create noisy pre-work interview and assessment signals for every candidate:

```bash
.venv/bin/python ground-truth-generation/scripts/generate_candidate_assessments.py
```

The notebook at `notebooks/labeling_sanity_check.ipynb` contains the same checks
plus charts, example candidate-job pairs, and plain-language conclusions. Its
outputs are intentionally cleared before commit so generated data stays local.
