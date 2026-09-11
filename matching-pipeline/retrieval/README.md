# Retrieval

Retrieval has one job: for each job, cheaply reduce the full candidate pool to a
small set that contains as many relevant candidates as possible. It does not make
the final hiring or allocation decision.

The implementation plan is in [`PLAN.md`](PLAN.md).

## Current implementation

The first increment contains typed leakage-safe schemas, deterministic candidate
and job renderers, exact cosine search, Recall@K, and a thin LangChain
`PGVectorStore` adapter. It intentionally does not send data to an embedding API.

Run the unit tests from the repository root:

```bash
PYTHONPATH=matching-pipeline/retrieval/src \
  .venv/bin/python -m unittest discover \
  -s matching-pipeline/retrieval/tests -v
```

Start a local pgvector database when the index integration is ready to be exercised:

```bash
docker compose -f matching-pipeline/retrieval/docker-compose.yml up -d
```

The local connection string is:

```text
postgresql+asyncpg://retrieval:retrieval@localhost:6024/retrieval
```

Create resumable embedding artifacts (use `--limit` for an inexpensive pilot):

```bash
PYTHONPATH=matching-pipeline/retrieval/src \
  .venv/bin/python matching-pipeline/retrieval/scripts/embed_profiles.py \
  candidates --limit 100

PYTHONPATH=matching-pipeline/retrieval/src \
  .venv/bin/python matching-pipeline/retrieval/scripts/embed_profiles.py \
  jobs --limit 10
```

Run the exact-search reference and write retrieval and Recall@K artifacts:

```bash
PYTHONPATH=matching-pipeline/retrieval/src \
  .venv/bin/python matching-pipeline/retrieval/scripts/evaluate_exact.py \
  --ks 10 25 50 100
```

Generated artifacts are local and gitignored. A limited pilot evaluates labels
only among candidates that were actually searchable; its metrics validate the
pipeline but are not comparable to a full-pool result.

Load cached vectors into pgvector and build the serving index without re-embedding:

```bash
export RETRIEVAL_DATABASE_URL=postgresql+asyncpg://retrieval:retrieval@localhost:6024/retrieval
PYTHONPATH=matching-pipeline/retrieval/src \
  .venv/bin/python matching-pipeline/retrieval/scripts/index_hnsw.py --overwrite
```

`--overwrite` is appropriate for a reproducible local rebuild and is never implied.
Then compare HNSW results and latency with the exact reference:

```bash
PYTHONPATH=matching-pipeline/retrieval/src \
  .venv/bin/python matching-pipeline/retrieval/scripts/evaluate_hnsw.py \
  --ks 10 25 50 100
```

HNSW is treated only as a scalable serving mechanism. Exact/HNSW overlap checks
that approximation has not materially changed results; relevance Recall@K remains
the measure of the representation and embedding model. Authentic-resume profiles
can be appended once their separate extraction work is complete.

Explore Recall@K for a reproducible random sample of jobs in
[`notebooks/recall_at_k_sample.ipynb`](notebooks/recall_at_k_sample.ipynb). Change
`X` in the parameter cell to control the sample size, then run the notebook from
the repository root.

Use [`notebooks/retrieve_top_5_for_job.ipynb`](notebooks/retrieve_top_5_for_job.ipynb)
to paste in a new job description and inspect the five nearest candidates from
the HNSW index. Start the local pgvector service before running it.
