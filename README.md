# Marketplace ranking system

The repository is organized as independent pipeline stages. The first stage is
[data acquisition](data-acquisition/README.md), which contains the authentic-data
downloads, cleaning, EDA, resume-population planning, and synthetic-resume pilot.

```text
marketplace-ranking-system/
├── data-acquisition/   # Current self-contained pipeline stage
├── .venv/              # Shared local Python environment
├── .env                # Local project settings and secrets
└── .gitignore
```

Run commands from the repository root. Detailed commands and data documentation
are in `data-acquisition/README.md`.

Relevance labels and simulated performance outcomes are built in
[`ground-truth-generation/`](ground-truth-generation/).

The downstream matching work is organized in [`matching-pipeline/`](matching-pipeline/):
retrieval produces a high-recall candidate set, modeling reranks that set, and
allocation applies marketplace-wide capacity and business constraints.
