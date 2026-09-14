# Matching pipeline

This stage turns normalized candidates and jobs into marketplace decisions. It is
split at the boundaries that would exist in a production multi-stage matcher:

```text
job + candidate pool
        |
        v
retrieval/  -> high-recall candidate shortlist
        |
        v
modeling/   -> pairwise scores and ranked candidates per job
        |
        v
allocation/ -> assignments subject to capacity and business constraints
```

- [`retrieval/`](retrieval/README.md) contains the current implementation plan.
- [`modeling/`](modeling/README.md) is reserved for feature generation, learned
  reranking, and ranking evaluation.
- [`allocation/`](allocation/README.md) contains binary linear optimization for
  globally assigning ranked candidates across competing jobs.

Each subdirectory should remain independently testable. Retrieval owns candidate
IDs and similarity scores, modeling owns candidate-job scores, and allocation owns
final decisions. Labels from `ground-truth-generation/data/v1/` are evaluation
data; hidden simulator fields are never ranking inputs.

Run the connected retrieval and modeling stages for a stored job:

```bash
PYTHONPATH=matching-pipeline/retrieval/src:matching-pipeline/retrieval/scripts:matching-pipeline/modeling/src \
  .venv/bin/python matching-pipeline/scripts/rank_job.py --job-id JOB_ID
```

The command retrieves 50 candidates by default, reranks that shortlist with the
saved LightGBM model, and returns the top five with both retrieval similarity and
predicted performance probability. Pass `--backend hnsw` to use pgvector.

For the complete multi-job walkthrough with visualizations, open
[`notebooks/marketplace_walkthrough.ipynb`](notebooks/marketplace_walkthrough.ipynb).
It includes a reproducible offline demo and a live mode for pasted job descriptions.
