"""Regenerate only performance outcomes from existing relevance and hidden tables."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from generate_ground_truth import DEFAULT_CONFIG, DEFAULT_OUTPUT, outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    candidate_hidden = pd.read_parquet(args.data_dir / "candidate_hidden.parquet")
    job_hidden = pd.read_parquet(args.data_dir / "job_hidden.parquet")
    count = outcomes(args.data_dir, candidate_hidden, job_hidden, config["seed"])
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outcome_pairs"] = count
    manifest["outcome_rng_version"] = "independent_hash_streams_v2"
    manifest["outcomes_regenerated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outcome_pairs": count, "outcome_rng_version": manifest["outcome_rng_version"]}, indent=2))


if __name__ == "__main__":
    main()
