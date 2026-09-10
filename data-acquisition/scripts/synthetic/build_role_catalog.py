"""Build role-archetype requests from authentic job descriptions and collect results."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from tqdm import tqdm

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ACQUISITION_ROOT.parent
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.authentic.load_data import authentic_data_directory, data_directory, file_hash
from scripts.synthetic.io_utils import (
    read_jsonl, read_jsonl_checkpoint, response_output_text, successful_batch_bodies, write_jsonl,
)
from scripts.synthetic.models import RoleArchetype, strict_schema

VERSION = "v1"
LOG = logging.getLogger(__name__)
DEFAULT_OUTPUT = data_directory() / f"synthetic/generated/role_catalog/{VERSION}"

INSTRUCTIONS = """You build realistic occupation archetypes for synthetic English resumes.
Use patterns shared across the supplied historical job descriptions. Do not copy an employer,
location, compensation, personal detail, or a sentence from any posting. Keep skills concrete
and occupation-specific. Education and certifications must be plausible options, not universal
requirements. Return the exact supplied role_family and role_type. Divide a broad role_type into
coherent career tracks when its titles represent materially different work. For example, keep
corporate financial analysis separate from personal financial advising, and keep tax work separate
from accounts receivable. Every track's titles, skills, responsibilities, education, certifications,
previous titles, and industries must describe that same career. Use short noun phrases."""


def request_body(model: str, role_family: str, role_type: str, common_titles: list[str], examples: list[dict]) -> dict:
    source = {
        "role_family": role_family,
        "role_type": role_type,
        "common_titles": common_titles,
        "job_examples": examples,
    }
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
        ],
        "text": {"format": {
            "type": "json_schema", "name": "role_archetype", "strict": True,
            "schema": strict_schema(RoleArchetype),
        }},
    }


def prepare(output_dir: Path, jobs_per_role: int, seed: int, limit: int | None) -> dict:
    if not 2 <= jobs_per_role <= 20:
        raise ValueError("jobs-per-role must be between 2 and 20")
    load_dotenv(PROJECT_ROOT / ".env")
    model = os.getenv("OPENAI_CATALOG_MODEL", "gpt-5.6-terra")
    authentic = authentic_data_directory()
    plan_path = authentic / "reports/role_families/resume_generation_plan.csv"
    mapping_path = authentic / "processed/job_role_mapping.parquet"
    jobs_path = authentic / "processed/jobs.parquet"
    plan = pd.read_csv(plan_path).sort_values(["role_family", "role_type"])
    if limit:
        plan = plan.head(limit)
    mapping = pd.read_parquet(mapping_path, columns=["job_id", "role_family", "role_type"])
    jobs = pd.read_parquet(jobs_path, columns=["job_id", "title", "description", "formatted_experience_level"])
    joined = mapping.merge(jobs, on="job_id", validate="one_to_one")
    requests, sources = [], []
    for row_number, row in enumerate(tqdm(plan.itertuples(index=False), total=len(plan),
                                          desc="Catalog requests", unit="role", dynamic_ncols=True), start=1):
        group = joined.loc[
            joined.role_family.eq(row.role_family) & joined.role_type.eq(row.role_type)
        ].copy()
        common_titles = str(row.example_titles).split(" | ")
        common_count = min(len(common_titles), max(1, jobs_per_role // 2))
        common_rows = pd.concat([
            group.loc[group.title.eq(title)].sort_values("job_id").head(1)
            for title in common_titles[:common_count]
        ], ignore_index=True).drop_duplicates("job_id")
        remaining = group.loc[~group.job_id.isin(common_rows.job_id)]
        random_rows = remaining.sample(
            min(jobs_per_role - len(common_rows), len(remaining)), random_state=seed + row_number
        )
        sample = pd.concat([common_rows, random_rows], ignore_index=True)
        examples = [{
            "job_id": str(item.job_id),
            "title": str(item.title),
            "experience_level": None if pd.isna(item.formatted_experience_level) else str(item.formatted_experience_level),
            "description_excerpt": str(item.description)[:2200],
        } for item in sample.itertuples(index=False)]
        custom_id = f"role-{row_number:03d}"
        requests.append({"custom_id": custom_id, "method": "POST", "url": "/v1/responses",
                         "body": request_body(model, row.role_family, row.role_type, common_titles, examples)})
        sources.append({"custom_id": custom_id, "role_family": row.role_family,
                        "role_type": row.role_type, "source_job_ids": [e["job_id"] for e in examples]})
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "requests.jsonl"
    source_path = output_dir / "request_sources.jsonl"
    write_jsonl(request_path, requests)
    write_jsonl(source_path, sources)
    manifest = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "seed": seed, "jobs_per_role": jobs_per_role,
        "request_count": len(requests), "requests_sha256": file_hash(request_path),
        "source_plan_sha256": file_hash(plan_path),
    }
    (output_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def collect(output_dir: Path, responses_path: Path) -> dict:
    sources = {item["custom_id"]: item for item in read_jsonl(output_dir / "request_sources.jsonl")}
    bodies = successful_batch_bodies(responses_path)
    records = []
    for custom_id, source in sources.items():
        if custom_id not in bodies:
            raise ValueError(f"Missing batch response: {custom_id}")
        archetype = RoleArchetype.model_validate_json(response_output_text(bodies[custom_id]))
        if (archetype.role_family, archetype.role_type) != (source["role_family"], source["role_type"]):
            raise ValueError(f"Role identity changed for {custom_id}")
        records.append({**archetype.model_dump(), "source_job_ids": source["source_job_ids"]})
    catalog_path = output_dir / "role_archetypes.jsonl"
    write_jsonl(catalog_path, records)
    manifest = {"version": VERSION, "record_count": len(records), "responses_sha256": file_hash(responses_path),
                "catalog_sha256": file_hash(catalog_path)}
    (output_dir / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def direct(output_dir: Path, jobs_per_role: int, seed: int, limit: int | None,
           workers: int = 8, restart: bool = False) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    manifest = prepare(output_dir, jobs_per_role, seed, limit)
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the project-root .env")
    requests = read_jsonl(output_dir / "requests.jsonl")
    checkpoint_path = output_dir / "direct_checkpoint.jsonl"
    checkpoint_meta_path = output_dir / "direct_checkpoint.json"
    checkpoint_meta = {
        "requests_sha256": manifest["requests_sha256"], "model": manifest["model"],
        "request_count": manifest["request_count"],
    }
    existing = {}
    if not restart and checkpoint_meta_path.exists() and checkpoint_path.exists():
        if json.loads(checkpoint_meta_path.read_text()) == checkpoint_meta:
            existing = {item["custom_id"]: item for item in read_jsonl_checkpoint(checkpoint_path)}
    if not existing:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("")
        checkpoint_meta_path.write_text(json.dumps(checkpoint_meta, indent=2) + "\n")

    pending = [request for request in requests if request["custom_id"] not in existing]
    sources = {item["custom_id"]: item for item in read_jsonl(output_dir / "request_sources.jsonl")}
    local = threading.local()

    def send(request: dict) -> dict:
        if not hasattr(local, "client"):
            local.client = OpenAI()
        source = sources[request["custom_id"]]
        body = copy.deepcopy(request["body"])
        for attempt in range(1, 5):
            response = local.client.responses.create(**body)
            try:
                archetype = RoleArchetype.model_validate_json(response.output_text)
                if (archetype.role_family, archetype.role_type) != (source["role_family"], source["role_type"]):
                    raise ValueError(f"Role identity changed for {request['custom_id']}")
                return {"custom_id": request["custom_id"], "response": {
                    "status_code": 200, "body": response.model_dump()}, "error": None}
            except Exception as error:
                if attempt == 4:
                    raise
                LOG.warning("Retrying %s after attempt %s failed validation: %s",
                            request["custom_id"], attempt, error)
                body["input"].append({
                    "role": "developer",
                    "content": f"The previous result failed validation. Correct it exactly: {str(error)[:400]}",
                })
                time.sleep(attempt)
        raise RuntimeError(f"Unable to generate {request['custom_id']}")

    progress = tqdm(total=len(requests), initial=len(existing), desc=f"Role archetypes ({workers} workers)",
                    unit="role", dynamic_ncols=True)
    failures, fatal_errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending_iterator = iter(pending)
        futures = {}

        def submit_next() -> bool:
            try:
                request = next(pending_iterator)
            except StopIteration:
                return False
            futures[executor.submit(send, request)] = request["custom_id"]
            return True

        for _ in range(min(workers, len(pending))):
            submit_next()
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                custom_id = futures.pop(future)
                try:
                    line = future.result()
                except Exception as error:
                    failures.append({"custom_id": custom_id, "error": str(error)})
                    LOG.error("Failed %s after all retries: %s", custom_id, error)
                    if isinstance(error, OpenAIError):
                        fatal_errors.append(error)
                else:
                    existing[line["custom_id"]] = line
                    with checkpoint_path.open("a") as handle:
                        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                progress.update(1)
            if not fatal_errors:
                for _ in completed:
                    submit_next()
    progress.close()
    failure_path = output_dir / "direct_failures.jsonl"
    write_jsonl(failure_path, failures)
    if fatal_errors:
        raise RuntimeError(
            f"OpenAI API error stopped new submissions. Completed roles were checkpointed; "
            f"fix the API error and rerun. Details: {failure_path}"
        ) from fatal_errors[0]
    if failures:
        raise RuntimeError(
            f"{len(failures)} role(s) failed after four attempts. Successful roles were checkpointed; "
            f"rerun to retry only failures. Details: {failure_path}"
        )
    lines = [existing[request["custom_id"]] for request in requests]
    response_path = output_dir / "direct_responses.jsonl"
    write_jsonl(response_path, lines)
    return collect(output_dir, response_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--jobs-per-role", type=int, default=8)
    prep.add_argument("--seed", type=int, default=20260910)
    prep.add_argument("--limit", type=int)
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--responses", type=Path, required=True)
    collect_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    direct_parser = sub.add_parser("direct")
    direct_parser.add_argument("--jobs-per-role", type=int, default=8)
    direct_parser.add_argument("--seed", type=int, default=20260910)
    direct_parser.add_argument("--limit", type=int, default=3)
    direct_parser.add_argument("--workers", type=int, default=8)
    direct_parser.add_argument("--restart", action="store_true",
                               help="Ignore the compatible checkpoint and regenerate every role")
    direct_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if args.command == "prepare":
        result = prepare(args.output_dir, args.jobs_per_role, args.seed, args.limit)
    elif args.command == "collect":
        result = collect(args.output_dir, args.responses)
    else:
        result = direct(args.output_dir, args.jobs_per_role, args.seed, args.limit, args.workers, args.restart)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
