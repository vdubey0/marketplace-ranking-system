"""Extract anonymized structured candidate profiles from authentic resume examples."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import Field
from tqdm import tqdm

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ACQUISITION_ROOT.parent
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.authentic.load_data import data_directory, file_hash
from scripts.synthetic.io_utils import read_jsonl_checkpoint, write_jsonl
from scripts.synthetic.models import StrictModel, strict_schema


VERSION = "v1"
LOG = logging.getLogger(__name__)
DEFAULT_INPUT = data_directory() / "authentic/processed/resumes.parquet"
DEFAULT_OUTPUT = data_directory() / f"authentic/processed/candidate_profiles_{VERSION}"
ROLE_CATALOG = data_directory() / "synthetic/generated/role_catalog/v1/role_archetypes.jsonl"


class AuthenticExtraction(StrictModel):
    resume_id: str
    sanitized_resume_text: str = Field(min_length=100)
    current_title: str
    skills: list[str] = Field(min_length=1, max_length=30)
    years_experience: float = Field(ge=0, le=50)
    education_level: Literal["high_school", "associates", "bachelors", "masters", "doctorate", "other_or_unspecified"]
    industries: list[str] = Field(max_length=10)
    certifications: list[str] = Field(max_length=15)
    role_family: str
    role_type: str
    role_track: str
    seniority: Literal["entry", "mid", "senior", "leadership"]


def catalog_options() -> tuple[list[dict], set[tuple[str, str, str]]]:
    rows = [json.loads(line) for line in ROLE_CATALOG.read_text().splitlines() if line.strip()]
    options, allowed = [], set()
    for row in rows:
        for track in row["tracks"]:
            item = {"role_family": row["role_family"], "role_type": row["role_type"],
                    "role_track": track["track_name"], "example_titles": track["target_titles"]}
            options.append(item)
            allowed.add((item["role_family"], item["role_type"], item["role_track"]))
    return options, allowed


def scrub_obvious_pii(text: str) -> str:
    """Remove contact strings before the model sees them; the model removes names/addresses."""
    text = re.sub(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", "[EMAIL REMOVED]", text)
    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+", "[URL REMOVED]", text)
    text = re.sub(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)", "[PHONE REMOVED]", text)
    return text


def request_body(model: str, resume_id: str, category: str, resume_text: str, options: list[dict]) -> dict:
    instructions = """Extract a candidate profile from the provided English resume.
Use only facts stated in the resume. Do not invent skills, credentials, industries,
titles, or experience. Choose exactly one supplied role option. Return resume_id
unchanged. In sanitized_resume_text, preserve job-relevant content but remove or
replace personal names, email addresses, phone numbers, street addresses, precise
locations, personal URLs, social profile URLs, and other contact details. Employer
and school names may remain. Do not insert a replacement person's name. Estimate
years_experience conservatively from stated dates. Seniority describes the candidate's
most recent relevant work, not simply the number of years worked."""
    payload = {"resume_id": resume_id, "source_category": category,
               "allowed_role_options": options, "resume_text": scrub_obvious_pii(resume_text)}
    return {"model": model, "store": False, "reasoning": {"effort": "low"},
            "input": [{"role": "developer", "content": instructions},
                      {"role": "user", "content": json.dumps(payload)}],
            "text": {"format": {"type": "json_schema", "name": "authentic_candidate_profile",
                                  "strict": True, "schema": strict_schema(AuthenticExtraction)}}}


def validate(item: AuthenticExtraction, expected_id: str, allowed: set[tuple[str, str, str]]) -> None:
    if item.resume_id != expected_id:
        raise ValueError(f"resume_id changed: expected {expected_id}, got {item.resume_id}")
    role = (item.role_family, item.role_type, item.role_track)
    if role not in allowed:
        raise ValueError(f"role selection is not in the supplied catalog: {role}")
    text = item.sanitized_resume_text
    forbidden = [r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", r"(?i)\b(?:https?://|www\.)\S+",
                 r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"]
    if any(re.search(pattern, text) for pattern in forbidden):
        raise ValueError("sanitized_resume_text still contains contact information")


def extraction_hash(source_hash: str, model: str) -> str:
    return hashlib.sha256(f"{VERSION}|{source_hash}|{model}".encode()).hexdigest()


def run(input_path: Path, output_dir: Path, limit: int | None, workers: int, restart: bool) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the project-root .env")
    model = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-5.6-luna")
    source = pd.read_parquet(input_path, columns=["resume_id", "resume_text", "category", "text_sha256"])
    if limit is not None:
        source = source.head(limit)
    options, allowed = catalog_options()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "extractions_checkpoint.jsonl"
    metadata_path = output_dir / "checkpoint_metadata.json"
    source_hash = file_hash(input_path)
    metadata = {"version": VERSION, "source_sha256": source_hash, "model": model,
                "record_count": len(source), "extraction_hash": extraction_hash(source_hash, model)}
    existing: dict[str, AuthenticExtraction] = {}
    if not restart and checkpoint.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == metadata:
        existing = {x["resume_id"]: AuthenticExtraction.model_validate(x) for x in read_jsonl_checkpoint(checkpoint)}
    else:
        checkpoint.write_text("")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    records = source.to_dict("records")
    pending = [x for x in records if str(x["resume_id"]) not in existing]
    local = threading.local()

    def send(record: dict) -> AuthenticExtraction:
        if not hasattr(local, "client"):
            # Bound network stalls so a worker retries instead of hanging for the
            # SDK's long default timeout. Outer retries still provide four chances.
            local.client = OpenAI(timeout=60.0, max_retries=1)
        resume_id = str(record["resume_id"])
        body = request_body(model, resume_id, str(record["category"]), str(record["resume_text"]), options)
        for attempt in range(1, 5):
            try:
                result = AuthenticExtraction.model_validate_json(local.client.responses.create(**body).output_text)
                validate(result, resume_id, allowed)
                return result
            except Exception as error:
                if attempt == 4:
                    raise
                LOG.warning("Retrying %s after attempt %s: %s", resume_id, attempt, error)
                body["input"].append({"role": "developer", "content": f"Correct the prior response. Validation error: {str(error)[:400]}"})
                time.sleep(attempt)
        raise RuntimeError(resume_id)

    failures, fatal = [], []
    progress = tqdm(total=len(records), initial=len(existing), desc=f"Authentic profile extraction ({workers} workers)", unit="resume", dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator, futures = iter(pending), {}
        def submit() -> bool:
            try: record = next(iterator)
            except StopIteration: return False
            futures[executor.submit(send, record)] = str(record["resume_id"]); return True
        for _ in range(min(workers, len(pending))): submit()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                resume_id = futures.pop(future)
                try: item = future.result()
                except Exception as error:
                    failures.append({"resume_id": resume_id, "error": str(error)})
                    LOG.error("Failed %s: %s", resume_id, error)
                    if isinstance(error, OpenAIError): fatal.append(error)
                else:
                    existing[item.resume_id] = item
                    with checkpoint.open("a") as handle: handle.write(item.model_dump_json() + "\n")
                progress.update(1)
            if not fatal:
                for _ in done: submit()
    progress.close()
    write_jsonl(output_dir / "failures.jsonl", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} extraction(s) failed; rerun to retry them. See {output_dir/'failures.jsonl'}")
    ordered = [existing[str(x)] for x in source.resume_id]
    rows=[]
    for number, (raw, item) in enumerate(zip(records, ordered), 1):
        rows.append({"candidate_id": f"auth_{VERSION}_{number:06d}", "source_resume_id": item.resume_id,
                     "name": f"Authentic Candidate {number:06d}", "resume_text": item.sanitized_resume_text,
                     "current_title": item.current_title, "skills": item.skills,
                     "years_experience": item.years_experience, "education_level": item.education_level,
                     "industries": item.industries, "domain": "unknown", "role_family": item.role_family,
                     "role_type": item.role_type, "role_track": item.role_track, "seniority": item.seniority,
                     "certifications": item.certifications, "is_synthetic": False, "fit_profile": None,
                     "label_status": "labeled", "source": "authentic_resume_example",
                     "source_text_sha256": raw["text_sha256"]})
    table = pd.DataFrame(rows)
    table_path = output_dir / "candidates.parquet"
    table.to_parquet(table_path, index=False)
    manifest = {**metadata, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_count": len(table), "table_sha256": file_hash(table_path),
                "pii_policy": "LLM-sanitized; contact-pattern validation applied"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpcore").setLevel(logging.WARNING)
    print(json.dumps(run(args.input, args.output_dir, args.limit, args.workers, args.restart), indent=2))


if __name__ == "__main__": main()
