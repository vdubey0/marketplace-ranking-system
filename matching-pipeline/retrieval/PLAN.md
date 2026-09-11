# Vector retrieval plan

## 1. Objective and boundary

Given a job and the candidates eligible at query time, return the top `K`
candidate IDs and retrieval scores. Optimize relevance Recall@K while recording
latency and shortlist size. The later modeling stage can remove false positives;
a relevant candidate omitted here cannot be recovered later.

Version 1 deliberately stops at semantic retrieval. Structured filters, hybrid
BM25 retrieval, alternate embedding models, and embedding fine-tuning are
experiments to run against a fixed baseline, not requirements for the first pass.

## 2. Inputs and leakage rules

Use these public tables from `ground-truth-generation/data/v1/`:

- `candidates.parquet`: `candidate_id`, resume text, title, skills, experience,
  education, industries, domain, role taxonomy, seniority, and certifications.
- `jobs.parquet`: `job_id`, title, description, role taxonomy, seniority, minimum
  experience, required/preferred skills, certifications, and domain.
- `relevance_ground_truth.parquet`: relevance grades 1 and 2 for evaluation only.
- `relevance_negative_sample.parquet`: sampled grade-0 pairs for diagnostics only;
  it is not a complete set of negatives.

Never put names, `fit_profile`, `is_synthetic`, `relevance_grade`, relevance
component scores, outcomes, or any field from `candidate_hidden.parquet` or
`job_hidden.parquet` into text, metadata used for scoring, or filters. Candidate
names are identifiers in disguise and are excluded. The synthetic dataset already
contains extracted resume text and structured fields, so the first pass consumes
those fields directly; a raw PDF extraction adapter can be added later without
changing the retrieval interface.

## 3. Canonical representations

Create deterministic, versioned renderers rather than embedding raw JSON or an
LLM's unconstrained prose.

Candidate embedding text:

```text
Current title: ...
Role: <family> | <type> | <track>
Seniority: ...
Experience: ... years
Skills: ...
Industries and domain: ...
Education: ...
Certifications: ...
Resume: ...
```

Job embedding text:

```text
Title: ...
Role: <family> | <type> | <track>
Seniority: ...
Minimum experience: ... years
Required skills: ...
Preferred skills: ...
Domain: ...
Required certifications: ...
Description: ...
```

Normalize whitespace, preserve meaningful tokens such as `C++` and `.NET`, sort
set-like fields, label missing values explicitly, and cap text using a documented
field-priority policy rather than blind tail truncation. Store the renderer
version and a SHA-256 content hash. If LLM parsing is later introduced, validate
its JSON against a schema, retain the source text, record prompt/model versions,
and treat absent facts as unknown rather than inferred.

## 4. Embeddings and storage

Start with one shared text-embedding model for candidates and jobs and cosine
distance on normalized vectors. Record the model name/version, vector dimension,
normalization, creation time, and content hash so unchanged candidates are not
re-embedded.

Use Postgres with `pgvector` as the first vector database because it supports HNSW,
metadata filters, reproducible local setup, and matches the intended production
stack. Keep storage behind a small interface (`upsert_candidates`, `delete`,
`search`) so another ANN backend can be compared later.

Minimum vector record:

| Field | Purpose |
| --- | --- |
| `candidate_id` | Stable primary key |
| `embedding` | Candidate vector |
| `embedding_model` | Model/version provenance |
| `representation_version` | Renderer provenance |
| `content_hash` | Idempotent refreshes |
| `indexed_at` | Operational auditability |
| eligible metadata | Query-time hard filters only |

Build an HNSW cosine index. Treat construction parameters and query search depth
as configuration, not constants embedded in code. Persist index size and build
time alongside experiment results.

## 5. Retrieval path

1. Validate and render the job with the versioned job renderer.
2. Embed it with the exact model used for candidate vectors.
3. Apply only genuine eligibility constraints before ANN search (for example a
   legally required certification). Do not initially use role family, seniority,
   or skill match as hard filters because those can silently destroy recall.
4. Query the HNSW index for at least the largest evaluated `K`.
5. Return `job_id`, `candidate_id`, rank, cosine similarity, model/version fields,
   filter configuration, and latency.

If a metadata filter produces too few candidates, log the eligible-pool size and
use a documented fallback rather than silently returning a short list.

## 6. Evaluation design

Evaluate against `relevance_ground_truth.parquet`; grade 1 and grade 2 are relevant
for the primary metric. Also report a stricter grade-2-only view. For each job
`j`:

```text
Recall@K(j) = |retrieved_K(j) intersect relevant(j)| / |relevant(j)|
```

Track `K = 10, 25, 50, 100, 250, 500` initially, bounded by the eligible pool.
Report:

- per-job Recall@K (the auditable base table);
- macro Recall@K, so large jobs do not dominate;
- micro Recall@K, to describe total relevant-pair coverage;
- median, 10th percentile, and worst-job recall;
- number of evaluable jobs and relevant candidates per job;
- p50/p95 query latency and candidates returned;
- slices by role family, seniority, domain, and relevant-set size.

Jobs with zero labeled relevant candidates are not assigned a zero recall; exclude
them from Recall@K and report their count separately. Precision@K and NDCG@K are
useful diagnostics, but Recall@K is the retrieval selection metric.

Keep two distinct notions of recall:

- **relevance Recall@K**: whether known relevant candidates survive retrieval;
- **ANN recall@K**: overlap between HNSW results and exact cosine top-K results.

This separation diagnoses representation failures versus approximate-index
failures. Produce the exact-search baseline on this dataset before tuning HNSW.
HNSW cannot fix a representation whose exact neighbors are wrong.

## 7. Experiment sequence

Run controlled, reproducible experiments in this order:

1. Deterministic representations plus exact cosine search.
2. HNSW with no soft matching filters; compare ANN recall and latency to exact.
3. Tune HNSW construction/search parameters and candidate `K`.
4. Compare raw resume/job text with labeled structured representations.
5. Compare embedding models on the same job set and labels.
6. Test only defensible hard eligibility filters and measure lost positives.
7. Add a BM25 branch and union it with vector results; deduplicate and measure the
   recall/latency gain.
8. If error analysis justifies it, train with positive pairs plus role-confusable
   hard negatives, using job-disjoint train/validation/test splits.

Do not tune against every job and quote the same jobs as a final result. Freeze a
job-level test split early. A candidate may appear across splits, but no job may.

## 8. Error analysis

For missed relevant pairs, save the job representation, candidate representation,
exact rank, ANN rank (if present), similarity, labels, and slice metadata. Classify
each miss as one of:

- candidate absent because of a hard filter or stale/missing vector;
- present in exact top-K but missed by HNSW (index/search failure);
- outside exact top-K (representation/model failure);
- likely label or normalization issue discovered during review.

Review worst jobs and a fixed sample of successes and failures after each material
change. This is where decisions such as hybrid search, field weighting, or a
second vector should originate.

## 9. Deliverables and acceptance criteria

Implementation deliverables:

- typed schemas and deterministic candidate/job renderers;
- batch embedding with retries, caching, provenance, and cost/count logging;
- a local pgvector/HNSW index and idempotent indexing command;
- exact and HNSW retrieval commands with a common output schema;
- per-job metric output plus an aggregate report;
- unit tests for rendering, leakage exclusions, Recall@K, zero-positive jobs,
  deterministic ordering/ties, and filtered eligible pools;
- an experiment manifest containing dataset hash, split, code/config versions,
  embedding version, index settings, metrics, latency, and timestamp.

The baseline is complete when a clean checkout can build the index and reproduce
the exact and HNSW reports; every job has an inspectable retrieval file; ANN recall
and relevance recall are reported separately; and no forbidden field reaches the
retrieval features. Numerical targets should be set after measuring the exact
baseline, because choosing them beforehand would be arbitrary for this synthetic
corpus.

## 10. First implementation slice

Keep the first pull-sized slice small:

1. Define schemas, leakage allowlists, and renderers.
2. Add renderer and metric tests.
3. Generate embeddings for a small deterministic candidate/job sample.
4. Implement exact cosine retrieval and the per-job Recall@K artifact.
5. Inspect failures, then introduce pgvector/HNSW as the next slice.

This establishes whether the representation works before adding index complexity
and creates the exact baseline required to evaluate HNSW honestly.
