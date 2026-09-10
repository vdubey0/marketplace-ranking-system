# Data acquisition — English marketplace data

This first step collects public English resume examples and historical job
descriptions. It does not generate synthetic records or train a model.

The initial download contains **2,484 resume examples** and **123,849 job
postings**. After the English filter and removal of empty/repeated records,
the cleaned tables contain **2,481 resumes** and **107,250 job descriptions**.

## Directory layout

Each data-acquisition stage has its own script directory and data boundary:

```text
data-acquisition/
├── config/                 # Role-family matching rules
├── docs/                   # Generated plans and decisions
├── notebooks/              # EDA only
├── scripts/
│   ├── authentic/          # Download and clean source datasets
│   ├── planning/           # Decide the synthetic population mix
│   └── synthetic/          # Generate synthetic records and PDFs
├── tests/                  # Checks for each stage
└── data/
    ├── authentic/
    │   ├── raw/            # Unchanged downloaded archives
    │   ├── processed/      # Clean English Parquet tables
    │   └── reports/        # EDA and role-planning outputs
    └── synthetic/
        └── generated/      # Profiles, text, tables, and PDFs
```

This lets a later training or ranking stage consume `processed/` or `generated/`
without depending on the notebook.

## Open the notebook

The local `.venv` contains the installed packages. From this project directory:

```bash
source .venv/bin/activate
jupyter lab data-acquisition/notebooks/01_authentic_data_eda.ipynb
```

Select **Marketplace data (.venv)** as the notebook kernel. The notebook shows:

- Actual resume and job excerpts, plus an original resume PDF page.
- Downloaded and retained row counts, and reasons records were excluded.
- Resume categories and common job titles.
- Text lengths, missing fields, work types, and experience levels.
- Simple skill-word mention counts, clearly separated from skill extraction.

All reusable loading logic is in `data-acquisition/scripts/authentic/load_data.py`;
EDA stays in the notebook.

## Load data without EDA

```bash
.venv/bin/python data-acquisition/scripts/authentic/load_data.py
```

This downloads missing archives and writes English, deduplicated Parquet tables.
It reuses existing downloads. Language checks can take several minutes.

```bash
# Rebuild from downloaded archives with no network access:
.venv/bin/python data-acquisition/scripts/authentic/load_data.py --offline

# Explicitly replace archives with the current upstream versions:
.venv/bin/python data-acquisition/scripts/authentic/load_data.py --refresh
```

`DATA_DIR` and `ENGLISH_MIN_CONFIDENCE` are configured in `.env`. No API key is
required for these public download endpoints. `.env`, the environment, and data
files are ignored by Git. `.env.example` documents the settings.

## Recreate the environment

Tested with Python 3.13 on macOS. `data-acquisition/requirements.txt` lists direct
dependencies; `data-acquisition/requirements.lock.txt` records the complete
installed environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r data-acquisition/requirements.lock.txt
cp .env.example .env
.venv/bin/python -m ipykernel install --prefix .venv --name marketplace-data --display-name 'Marketplace data (.venv)'
.venv/bin/python data-acquisition/scripts/authentic/load_data.py
```

Use `data-acquisition/requirements.txt` for compatible version resolution on a
different platform.

## Plan synthetic resume types

```bash
.venv/bin/python data-acquisition/scripts/planning/plan_resumes.py --batch-size 1000
```

This groups existing job titles and proposes how many resumes to create. It does
not generate candidates or call an LLM. Read [the generation plan](docs/resume_generation_plan.md)
for counts and actual title examples. Matching rules are in
`data-acquisition/config/role_families.json`; per-job decisions and review lists
are saved under `data-acquisition/data/authentic/`.

Ambiguous and unmatched titles are kept for review and excluded from quotas.
Quantities follow the classified subset of our historical, deduplicated job data;
they do not estimate the broader labor market. Source resume-category counts are
only rough references, not verified counts of candidates for each detailed role.

## Generate the three-resume pilot

```bash
.venv/bin/python data-acquisition/scripts/synthetic/generate_resume_pilot.py
```

This creates one healthcare, one sales, and one accounting candidate because
those are the three families selected when the observed ratios are rounded to a
three-record batch. Each candidate has a structured profile, clean text, and a
one-page PDF with selectable text. See [the synthetic-data layout](data/synthetic/README.md).

The pilot is intentionally limited to three reviewed profiles. It will refuse a
larger count until the population generator is implemented and checked.

## Generate resumes at scale

The scalable generator has three data stages. Terra builds reusable role
archetypes from samples of authentic job descriptions, Python creates candidate
facts, and Luna writes only the summary and experience bullets.

Run a small synchronous preview:

```bash
.venv/bin/python data-acquisition/scripts/synthetic/build_role_catalog.py direct --limit 3 --jobs-per-role 4
.venv/bin/python data-acquisition/scripts/synthetic/generate_candidate_specs.py --count 10
.venv/bin/python data-acquisition/scripts/synthetic/render_resumes.py direct --limit 3
```

Run the complete synchronous pipeline for 1,000 resumes with one command:

```bash
.venv/bin/python data-acquisition/scripts/synthetic/run_resume_generator.py --count 1000 --workers 8
```

This command builds any missing role catalog, creates all specifications, calls
Luna, and writes the PDFs. Its progress bars show counts, elapsed time, processing
rate, and ETA. Role-archetype and resume-writing responses are checkpointed after every API call, so
rerunning the command can continue that stage after an interruption. Add
`--rebuild-catalog` only when you intentionally want new archetypes.

For a full run, prepare and submit the 48-role catalog as one Batch API job:

```bash
.venv/bin/python data-acquisition/scripts/synthetic/build_role_catalog.py prepare --jobs-per-role 8
.venv/bin/python data-acquisition/scripts/synthetic/openai_jobs.py submit \
  --input data-acquisition/data/synthetic/generated/role_catalog/v1/requests.jsonl \
  --state data-acquisition/data/synthetic/generated/role_catalog/v1/batch_state.json
.venv/bin/python data-acquisition/scripts/synthetic/openai_jobs.py wait \
  --state data-acquisition/data/synthetic/generated/role_catalog/v1/batch_state.json \
  --poll-seconds 15
.venv/bin/python data-acquisition/scripts/synthetic/openai_jobs.py download \
  --state data-acquisition/data/synthetic/generated/role_catalog/v1/batch_state.json \
  --output data-acquisition/data/synthetic/generated/role_catalog/v1/batch_responses.jsonl \
  --errors data-acquisition/data/synthetic/generated/role_catalog/v1/batch_errors.jsonl
.venv/bin/python data-acquisition/scripts/synthetic/build_role_catalog.py collect \
  --responses data-acquisition/data/synthetic/generated/role_catalog/v1/batch_responses.jsonl
```

Then create candidate specifications and submit their writing requests:

```bash
.venv/bin/python data-acquisition/scripts/synthetic/generate_candidate_specs.py --count 1000
.venv/bin/python data-acquisition/scripts/synthetic/render_resumes.py prepare
.venv/bin/python data-acquisition/scripts/synthetic/openai_jobs.py submit \
  --input data-acquisition/data/synthetic/generated/resumes/v1/requests/resume_requests.jsonl \
  --state data-acquisition/data/synthetic/generated/resumes/v1/manifests/batch_state.json
.venv/bin/python data-acquisition/scripts/synthetic/openai_jobs.py wait \
  --state data-acquisition/data/synthetic/generated/resumes/v1/manifests/batch_state.json \
  --poll-seconds 15
.venv/bin/python data-acquisition/scripts/synthetic/openai_jobs.py download \
  --state data-acquisition/data/synthetic/generated/resumes/v1/manifests/batch_state.json \
  --output data-acquisition/data/synthetic/generated/resumes/v1/responses/batch_responses.jsonl \
  --errors data-acquisition/data/synthetic/generated/resumes/v1/responses/batch_errors.jsonl
.venv/bin/python data-acquisition/scripts/synthetic/render_resumes.py collect \
  --responses data-acquisition/data/synthetic/generated/resumes/v1/responses/batch_responses.jsonl
```

The `wait` command logs completed requests, elapsed time, processing rate, and an
ETA. Collection
validates the returned IDs and prose, writes text and Parquet files, and produces
one-page PDFs locally. Synthetic provenance stays in JSON and Parquet fields; it
is excluded from resume text and PDFs so it cannot affect embeddings. API
responses use `store=false`.

## Checks

```bash
.venv/bin/python -m unittest discover -s data-acquisition/tests -v
.venv/bin/python -m pip check
```

See [the authentic-data documentation](data/authentic/README.md) for source attribution, field meanings,
cleaning rules, and limitations. Published resume examples are not independently
verified real-person records. These sources contain no measured hiring outcomes.
