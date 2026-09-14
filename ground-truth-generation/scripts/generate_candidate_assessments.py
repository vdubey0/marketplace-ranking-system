"""Create noisy, observable pre-work candidate assessment signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "ground-truth-generation/data/v1"
DEFAULT_SEED = 20260911


def generate(data_dir: Path, seed: int) -> pd.DataFrame:
    hidden = pd.read_parquet(data_dir / "candidate_hidden.parquet").sort_values("candidate_id")
    rng = np.random.default_rng(seed)

    # Interviews reflect broad ability, communication/reliability, and domain depth.
    interview = (
        0.55 * hidden.latent_ability.to_numpy()
        + 0.25 * hidden.latent_domain_depth.to_numpy()
        + 0.20 * hidden.latent_reliability.to_numpy()
        + rng.normal(0, 0.10, len(hidden))
    )
    # Assessments focus more directly on demonstrated skill mastery.
    assessment = (
        0.70 * hidden.latent_skill_mastery.to_numpy()
        + 0.30 * hidden.latent_ability.to_numpy()
        + rng.normal(0, 0.08, len(hidden))
    )
    result = pd.DataFrame({
        "candidate_id": hidden.candidate_id,
        "interview_score": np.clip(interview, 0, 1).round(6),
        "assessment_score": np.clip(assessment, 0, 1).round(6),
        "assessment_version": "v1",
    })
    result.to_parquet(data_dir / "candidate_assessments.parquet", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = generate(args.data_dir, args.seed)
    print(json.dumps({"candidates": len(result), "seed": args.seed,
                      "interview_mean": float(result.interview_score.mean()),
                      "assessment_mean": float(result.assessment_score.mean())}, indent=2))


if __name__ == "__main__":
    main()
