"""Prepare OpenAI resume-writing requests, collect them, and create local PDFs."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import html
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from tqdm import tqdm

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ACQUISITION_ROOT.parent
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.authentic.load_data import data_directory, file_hash
from scripts.synthetic.io_utils import (
    read_jsonl, read_jsonl_checkpoint, response_output_text, successful_batch_bodies, write_jsonl,
)
from scripts.synthetic.models import CandidateSpec, ResumeWriting, strict_schema

VERSION = "v1"
LOG = logging.getLogger(__name__)
DEFAULT_SPECS = data_directory() / "synthetic/generated/candidate_specs/v1/candidate_specs.jsonl"
DEFAULT_OUTPUT = data_directory() / f"synthetic/generated/resumes/{VERSION}"

INSTRUCTIONS = """Write concise English resume prose for the candidate specification.
The candidate specification is the only source of truth. Do not add or change skills,
credentials, certifications, employers, titles, industries, dates, responsibilities,
tools, team sizes, money, percentages, or other measurements. Do not use a name,
address, phone number, email, location, demographic trait, or protected characteristic.
Do not mention that the candidate, employer, credential, data, or resume is fictional,
synthetic, generated, a sample, or intended for research.
Write a two-sentence professional summary and 2-4 distinct bullets for every experience.
Bullets may paraphrase only that experience's listed responsibilities. Never copy the
job-posting style or write first-person prose. Return every experience_id exactly once."""


def request_body(model: str, spec: CandidateSpec) -> dict:
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": spec.model_dump_json()},
        ],
        "text": {"format": {
            "type": "json_schema", "name": "resume_writing", "strict": True,
            "schema": strict_schema(ResumeWriting),
        }},
    }


def validate_writing(spec: CandidateSpec, writing: ResumeWriting) -> None:
    if writing.candidate_id != spec.candidate_id:
        raise ValueError(f"Candidate ID changed: {spec.candidate_id}")
    expected = [item.experience_id for item in spec.experience]
    actual = [item.experience_id for item in writing.experience]
    if actual != expected:
        raise ValueError(f"Experience IDs changed for {spec.candidate_id}: {actual}")
    prose = " ".join([writing.professional_summary, *(bullet for item in writing.experience for bullet in item.bullets)])
    forbidden_labels = ("fictional", "synthetic", "not a real person", "research sample", "generated resume")
    if any(label in prose.casefold() for label in forbidden_labels):
        raise ValueError(f"Generation label leaked into resume prose for {spec.candidate_id}")
    specs_by_experience = {item.experience_id: item for item in spec.experience}
    shared_source = " ".join([spec.target_title, *spec.skills, *spec.certifications])
    for item in writing.experience:
        source = specs_by_experience[item.experience_id]
        allowed_numbers = set(re.findall(r"\d+(?:[.,]\d+)?[%+]?", " ".join([
            shared_source, source.title, *source.responsibilities,
        ])))
        for bullet in item.bullets:
            used_numbers = set(re.findall(r"\d+(?:[.,]\d+)?[%+]?", bullet))
            if not used_numbers.issubset(allowed_numbers):
                unexpected = sorted(used_numbers - allowed_numbers)
                raise ValueError(
                    f"Unspecified numeric claim in bullets for {spec.candidate_id}: {unexpected}"
                )


def prepare(specs_path: Path, output_dir: Path, limit: int | None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    model = os.getenv("OPENAI_RESUME_MODEL", "gpt-5.6-luna")
    specs = [CandidateSpec.model_validate(item) for item in read_jsonl(specs_path)]
    if limit:
        specs = specs[:limit]
    request_path = output_dir / "requests/resume_requests.jsonl"
    write_jsonl(request_path, ({"custom_id": spec.candidate_id, "method": "POST", "url": "/v1/responses",
                                "body": request_body(model, spec)} for spec in specs))
    manifest = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "request_count": len(specs), "specs_path": str(specs_path.resolve()),
        "specs_sha256": file_hash(specs_path), "requests_sha256": file_hash(request_path),
    }
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests/prepare.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def format_month(value: str) -> str:
    return datetime.strptime(value, "%Y-%m").strftime("%b %Y")


def clean_text(spec: CandidateSpec, writing: ResumeWriting) -> str:
    by_id = {item.experience_id: item for item in writing.experience}
    lines = [spec.target_title, "", "PROFESSIONAL SUMMARY", writing.professional_summary,
             "", "SKILLS", ", ".join(spec.skills), "", "EXPERIENCE"]
    for job in spec.experience:
        lines += [f"{job.title} — {job.company} | {format_month(job.start)}–{format_month(job.end)}"]
        lines += [f"• {bullet}" for bullet in by_id[job.experience_id].bullets]
    lines += ["", "EDUCATION"]
    lines += [f"{item.credential}, {item.institution}, {item.graduation_year}" for item in spec.education]
    if spec.certifications:
        lines += ["", "CERTIFICATIONS", *spec.certifications]
    return "\n".join(lines)


def recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ACQUISITION_ROOT))
    except ValueError:
        return str(path.resolve())


def normalized_pdf_text(value: str) -> str:
    """Normalize PDF extraction artifacts without changing visible resume content."""
    value = unicodedata.normalize("NFKC", value)
    # The renderer can wrap compound skills after punctuation, for example
    # ``full-\ncycle`` and ``lockout/\ntagout``. PDF extraction preserves the
    # newline even though the phrase is visibly continuous.
    value = re.sub(r"\s*([/&+\-])\s*", r"\1", value)
    return re.sub(r"\s+", " ", value).casefold()


def resume_html(spec: CandidateSpec, writing: ResumeWriting) -> str:
    esc = html.escape
    by_id = {item.experience_id: item for item in writing.experience}
    jobs = "".join(
        f"<section><table class='row'><tr><td><b>{esc(job.title)}</b></td>"
        f"<td class='dates'>{format_month(job.start)} – {format_month(job.end)}</td></tr></table>"
        f"<div class='muted'>{esc(job.company)} · {esc(job.industry)}</div><ul>"
        + "".join(f"<li>{esc(bullet)}</li>" for bullet in by_id[job.experience_id].bullets) + "</ul></section>"
        for job in spec.experience
    )
    education = "".join(f"<div><b>{esc(item.credential)}</b><br>{esc(item.institution)} · {item.graduation_year}</div>" for item in spec.education)
    certs = "" if not spec.certifications else "<h2>Certifications</h2><ul>" + "".join(f"<li>{esc(item)}</li>" for item in spec.certifications) + "</ul>"
    return f"""<html><head><style>
    * {{ box-sizing:border-box }} body {{ font-family:Arial;color:#20262b;font-size:9.2pt;line-height:1.3 }}
    h1 {{ color:#17324d;font:22pt Georgia;margin:4px 0 1px }} .id,.muted {{ color:#5c6770 }} .id {{ font-size:8pt;margin-bottom:10px }}
    h2 {{ color:#17324d;font:10pt Georgia;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #aab7c4;padding-bottom:3px;margin:10px 0 5px }}
    p {{ margin:0 0 4px }} ul {{ margin:3px 0 5px 17px;padding:0 }} li {{ margin-bottom:2px }}
    .row {{ width:100%;border-collapse:collapse }} .row td {{ padding:0 }}
    .row .dates {{ text-align:right;white-space:nowrap;font-size:8.7pt }}
    </style></head><body><h1>{esc(spec.target_title)}</h1>
    <h2>Professional Summary</h2><p>{esc(writing.professional_summary)}</p>
    <h2>Skills</h2><p class='muted'>{esc(' · '.join(spec.skills))}</p>
    <h2>Experience</h2>{jobs}<h2>Education</h2>{education}{certs}</body></html>"""


def write_pdf(spec: CandidateSpec, writing: ResumeWriting, path: Path) -> dict:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    spare, scale = page.insert_htmlbox(pymupdf.Rect(45, 25, 550, 805), resume_html(spec, writing), scale_low=.68)
    if spare < 0:
        document.close()
        raise ValueError(f"Resume does not fit one page: {spec.candidate_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    document.set_metadata({"title": spec.target_title, "subject": "Resume"})
    document.save(path, garbage=4, deflate=True)
    document.close()
    return {"bytes": path.stat().st_size, "sha256": file_hash(path), "render_scale": round(scale, 4)}


def materialize(specs_path: Path, output_dir: Path, writings: list[ResumeWriting]) -> dict:
    all_specs = {item["candidate_id"]: CandidateSpec.model_validate(item) for item in read_jsonl(specs_path)}
    text_dir, pdf_dir, profile_dir = output_dir / "text", output_dir / "pdf", output_dir / "profiles"
    records, checks = [], []
    for writing in tqdm(writings, desc="Resume files", unit="resume", dynamic_ncols=True):
        spec = all_specs.get(writing.candidate_id)
        if not spec:
            raise ValueError(f"No candidate spec for {writing.candidate_id}")
        validate_writing(spec, writing)
        text = clean_text(spec, writing)
        text_path = text_dir / f"{spec.candidate_id}.txt"
        profile_path = profile_dir / f"{spec.candidate_id}.json"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text + "\n")
        profile = {"spec": spec.model_dump(), "writing": writing.model_dump(), "resume_text": text}
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")
        pdf_path = pdf_dir / f"{spec.candidate_id}.pdf"
        pdf_meta = write_pdf(spec, writing, pdf_path)
        with pymupdf.open(pdf_path) as document:
            extracted = " ".join(page.get_text() for page in document)
        normalized_extracted = normalized_pdf_text(extracted)
        for phrase in [spec.target_title, *spec.skills]:
            if normalized_pdf_text(phrase) not in normalized_extracted:
                raise ValueError(f"PDF is missing {phrase!r} for {spec.candidate_id}")
        records.append({"candidate_id": spec.candidate_id, "is_synthetic": True,
                        "role_family": spec.role_family, "role_type": spec.role_type,
                        "role_track": spec.role_track,
                        "target_title": spec.target_title, "seniority": spec.seniority,
                        "fit_profile": spec.fit_profile, "resume_text": text,
                        "profile_path": recorded_path(profile_path),
                        "pdf_path": recorded_path(pdf_path)})
        checks.append({"candidate_id": spec.candidate_id, **pdf_meta})
    writings_path = output_dir / "responses/resume_writings.jsonl"
    write_jsonl(writings_path, (item.model_dump() for item in writings))
    table_path = output_dir / "resumes.parquet"
    pd.DataFrame(records).to_parquet(table_path, index=False)
    manifest = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "record_count": len(records), "specs_sha256": file_hash(specs_path),
                "writings_sha256": file_hash(writings_path),
                "table_sha256": file_hash(table_path), "pdf_checks": checks}
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests/generated.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def collect(specs_path: Path, output_dir: Path, responses_path: Path) -> dict:
    bodies = successful_batch_bodies(responses_path)
    writings = [ResumeWriting.model_validate_json(response_output_text(body)) for _, body in sorted(bodies.items())]
    return materialize(specs_path, output_dir, writings)


def direct(specs_path: Path, output_dir: Path, limit: int, workers: int = 8) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    prepared = prepare(specs_path, output_dir, limit)
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the project-root .env")
    requests = read_jsonl(output_dir / "requests/resume_requests.jsonl")
    checkpoint_path = output_dir / "responses/direct_checkpoint.jsonl"
    checkpoint_meta_path = output_dir / "responses/direct_checkpoint.json"
    checkpoint_meta = {
        "specs_sha256": prepared["specs_sha256"], "model": prepared["model"],
        "request_count": prepared["request_count"],
    }
    existing = {}
    if checkpoint_meta_path.exists() and checkpoint_path.exists():
        if json.loads(checkpoint_meta_path.read_text()) == checkpoint_meta:
            existing = {
                item["candidate_id"]: ResumeWriting.model_validate(item)
                for item in read_jsonl_checkpoint(checkpoint_path)
            }
    if not existing:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("")
        checkpoint_meta_path.write_text(json.dumps(checkpoint_meta, indent=2) + "\n")
    pending = [request for request in requests if request["custom_id"] not in existing]
    specs_by_id = {item["candidate_id"]: CandidateSpec.model_validate(item) for item in read_jsonl(specs_path)}
    local = threading.local()

    def send(request: dict) -> ResumeWriting:
        if not hasattr(local, "client"):
            local.client = OpenAI()
        expected_id = request["custom_id"]
        body = copy.deepcopy(request["body"])
        for attempt in range(1, 5):
            response = local.client.responses.create(**body)
            try:
                writing = ResumeWriting.model_validate_json(response.output_text)
                if writing.candidate_id != expected_id:
                    raise ValueError(f"Candidate ID changed: expected {expected_id}, got {writing.candidate_id}")
                validate_writing(specs_by_id[expected_id], writing)
                return writing
            except Exception as error:
                if attempt == 4:
                    raise
                LOG.warning("Retrying %s after attempt %s failed: %s", expected_id, attempt, error)
                body["input"].append({
                    "role": "developer",
                    "content": (
                        "The previous attempt failed validation. Try again and follow the candidate "
                        f"specification exactly. Validation error: {str(error)[:400]}"
                    ),
                })
                time.sleep(attempt)
        raise RuntimeError(f"Unable to generate {expected_id}")

    progress = tqdm(total=len(requests), initial=len(existing), desc=f"OpenAI resume writing ({workers} workers)",
                    unit="resume", dynamic_ncols=True)
    failures = []
    fatal_errors = []
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
                candidate_id = futures.pop(future)
                try:
                    writing = future.result()
                except Exception as error:
                    failures.append({"candidate_id": candidate_id, "error": str(error)})
                    LOG.error("Failed %s after all retries: %s", candidate_id, error)
                    if isinstance(error, OpenAIError):
                        fatal_errors.append(error)
                else:
                    existing[writing.candidate_id] = writing
                    with checkpoint_path.open("a") as handle:
                        handle.write(writing.model_dump_json() + "\n")
                progress.update(1)
            if not fatal_errors:
                for _ in completed:
                    submit_next()
    progress.close()
    failure_path = output_dir / "responses/direct_failures.jsonl"
    write_jsonl(failure_path, failures)
    if fatal_errors:
        raise RuntimeError(
            f"OpenAI API error stopped new submissions. Completed resumes were checkpointed; "
            f"fix the API error and rerun the same command. Details: {failure_path}"
        ) from fatal_errors[0]
    if failures:
        raise RuntimeError(
            f"{len(failures)} resume(s) failed after four attempts. Successful resumes were checkpointed; "
            f"rerun the same command to retry only the failures. Details: {failure_path}"
        )
    writings = [existing[request["custom_id"]] for request in requests]
    return materialize(specs_path, output_dir, writings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prep.add_argument("--limit", type=int)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    collect_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    collect_parser.add_argument("--responses", type=Path, required=True)
    direct_parser = sub.add_parser("direct")
    direct_parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    direct_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    direct_parser.add_argument("--limit", type=int, default=3)
    direct_parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if args.command == "prepare":
        result = prepare(args.specs, args.output_dir, args.limit)
    elif args.command == "collect":
        result = collect(args.specs, args.output_dir, args.responses)
    else:
        result = direct(args.specs, args.output_dir, args.limit, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
