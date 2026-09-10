"""Run the complete synchronous role-catalog and resume generator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.synthetic.build_role_catalog import DEFAULT_OUTPUT as CATALOG_OUTPUT, direct as build_catalog
from scripts.authentic.load_data import authentic_data_directory
from scripts.synthetic.generate_candidate_specs import DEFAULT_OUTPUT as SPECS_OUTPUT, generate as generate_specs
from scripts.synthetic.io_utils import read_jsonl
from scripts.synthetic.render_resumes import DEFAULT_OUTPUT as RESUME_OUTPUT, direct as render_resumes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--jobs-per-role", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent OpenAI requests in each API stage (default: 8)")
    parser.add_argument("--rebuild-catalog", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    catalog_path = CATALOG_OUTPUT / "role_archetypes.jsonl"
    plan = pd.read_csv(authentic_data_directory() / "reports/role_families/resume_generation_plan.csv")
    expected = set(zip(plan.role_family, plan.role_type))
    existing = set()
    if catalog_path.exists():
        existing = {(item["role_family"], item["role_type"]) for item in read_jsonl(catalog_path)}
    if args.rebuild_catalog or existing != expected:
        logging.info("Stage 1/3: building all %s role archetypes with Terra", len(expected))
        catalog = build_catalog(CATALOG_OUTPUT, args.jobs_per_role, args.seed, limit=None,
                                workers=args.workers, restart=args.rebuild_catalog)
        logging.info("Role catalog complete: %s roles", catalog["record_count"])
    else:
        logging.info("Stage 1/3: reusing complete %s-role catalog", len(existing))

    logging.info("Stage 2/3: generating %s candidate specifications", args.count)
    specs = generate_specs(catalog_path, SPECS_OUTPUT, args.count, args.seed)
    logging.info("Candidate specifications complete: %s", specs["record_count"])

    logging.info("Stage 3/3: writing resumes with Luna and creating PDFs")
    result = render_resumes(SPECS_OUTPUT / "candidate_specs.jsonl", RESUME_OUTPUT, args.count,
                            workers=args.workers)
    logging.info("Resume generation complete: %s", result["record_count"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
