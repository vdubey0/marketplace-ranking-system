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
- [`allocation/`](allocation/README.md) is reserved for global opportunity routing,
  exploration, and constraint-aware assignment.

Each subdirectory should remain independently testable. Retrieval owns candidate
IDs and similarity scores, modeling owns candidate-job scores, and allocation owns
final decisions. Labels from `ground-truth-generation/data/v1/` are evaluation
data; hidden simulator fields are never ranking inputs.
