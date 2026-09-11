#!/usr/bin/env python3
"""Create or resume candidate/job embedding artifacts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from retrieval.embeddings import embed_records
from retrieval.renderers import render_candidate, render_job
from retrieval.schemas import CandidateProfile, JobProfile

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "ground-truth-generation/data/v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/embeddings"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("candidates", "jobs"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not configured")
    source = pd.read_parquet(args.data_dir / f"{args.kind}.parquet")
    if args.limit is not None:
        source = source.sort_values(source.columns[0]).head(args.limit)
    if args.kind == "candidates":
        records = [(p.candidate_id, render_candidate(p))
                   for p in (CandidateProfile.from_row(row) for row in source.to_dict("records"))]
    else:
        records = [(p.job_id, render_job(p))
                   for p in (JobProfile.from_row(row) for row in source.to_dict("records"))]

    # Bound a single network stall; completed batches are already checkpointed.
    client = OpenAI(timeout=60.0, max_retries=3)
    def call(texts: list[str]):
        response = client.embeddings.create(
            model=args.model, input=texts, dimensions=args.dimensions, encoding_format="float"
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    path = args.output_dir / f"{args.kind}.parquet"
    result = embed_records(records, path, call, model=args.model,
                           dimensions=args.dimensions, batch_size=args.batch_size)
    print(f"Wrote {len(result):,} {args.kind} embeddings to {path}")


if __name__ == "__main__":
    main()
