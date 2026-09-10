"""Create the first three synthetic resume profiles, texts, and selectable PDFs.

This pilot uses the role-family ratios computed from authentic job descriptions.
It does not call an LLM. All resume prose is rendered from the structured facts
defined below, making profile-to-document consistency directly testable.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymupdf

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ACQUISITION_ROOT))
from scripts.authentic.load_data import authentic_data_directory, data_directory, file_hash
from scripts.planning.plan_resumes import allocate_counts

GENERATOR_VERSION = "pilot_v1"
DEFAULT_SEED = 20260910

# These facts are fictional. Skills and responsibilities were chosen after reading
# several authentic descriptions for each role, but no profile copies one person or job.
PILOT_PROFILES = {
    "Healthcare": {
        "role_type": "Nursing",
        "target_title": "Registered Nurse",
        "headline": "Registered Nurse | Ambulatory Care",
        "years_experience": 5,
        "summary": (
            "Registered nurse with five years of experience supporting adult patients in "
            "ambulatory and community-care settings. Experienced in patient assessment, "
            "telephone triage, care documentation, medication administration, and patient education."
        ),
        "skills": [
            "Patient assessment", "Telephone triage", "Electronic health records",
            "Medication administration", "Immunizations", "Patient education",
            "Care coordination", "Clinical documentation"
        ],
        "experience": [
            {
                "title": "Registered Nurse", "company": "Riverside Family Health",
                "start": "2024-09", "end": "2026-08",
                "bullets": [
                    "Assessed adult patients, recorded histories and vital signs, and prepared concise updates for providers.",
                    "Handled telephone triage using clinic protocols and documented calls in the electronic health record.",
                    "Administered medications and immunizations and explained follow-up instructions to patients.",
                    "Coordinated referrals and followed outstanding orders with the care team."
                ]
            },
            {
                "title": "Staff Nurse", "company": "Northgate Community Clinic",
                "start": "2021-06", "end": "2024-08",
                "bullets": [
                    "Provided routine nursing care in a busy outpatient clinic under established policies.",
                    "Educated patients about medication schedules, preventive screenings, and self-management.",
                    "Worked with medical assistants and providers to maintain accurate and timely records."
                ]
            }
        ],
        "education": [{"credential": "Bachelor of Science in Nursing", "institution": "Lakeshore State University", "year": 2021}],
        "certifications": ["Registered Nurse license", "Basic Life Support"],
        "source_job_ids": ["3900953690", "3904391436", "3885803153"],
        "template": "classic"
    },
    "Sales and business development": {
        "role_type": "Sales and account management",
        "target_title": "Account Executive",
        "headline": "Account Executive | Business-to-Business Sales",
        "years_experience": 4,
        "summary": (
            "Account executive with four years of experience building sales pipelines and managing "
            "business-to-business customer relationships. Comfortable with prospecting, discovery, "
            "proposal development, negotiation, forecasting, and CRM-based opportunity management."
        ),
        "skills": [
            "Prospecting", "Consultative selling", "Pipeline management", "CRM",
            "Sales forecasting", "Proposal development", "Negotiation", "Customer onboarding"
        ],
        "experience": [
            {
                "title": "Account Executive", "company": "Northstar Business Systems",
                "start": "2024-01", "end": "2026-08",
                "bullets": [
                    "Managed opportunities from initial outreach through discovery, proposal review, negotiation, and handoff.",
                    "Maintained account activity and forecasts in the CRM and reviewed pipeline changes with sales leadership.",
                    "Worked with implementation staff to set expectations and guide new customers through onboarding.",
                    "Prepared presentations that connected customer needs with practical service options."
                ]
            },
            {
                "title": "Sales Development Representative", "company": "Brightline Services Group",
                "start": "2022-01", "end": "2023-12",
                "bullets": [
                    "Researched prospective accounts and opened conversations through phone, email, and professional networks.",
                    "Qualified inbound and outbound leads and scheduled discovery meetings for account executives.",
                    "Kept contact details and follow-up tasks current in the CRM."
                ]
            }
        ],
        "education": [{"credential": "Bachelor of Business Administration", "institution": "Westbridge College", "year": 2021}],
        "certifications": [],
        "source_job_ids": ["3886896721", "3889774986", "3904958811"],
        "template": "modern"
    },
    "Accounting and finance": {
        "role_type": "Accounting, payroll and tax",
        "target_title": "Staff Accountant",
        "headline": "Staff Accountant | General Ledger and Reporting",
        "years_experience": 5,
        "summary": (
            "Staff accountant with five years of progressive experience in general accounting, "
            "account reconciliations, journal entries, billing, month-end close, and financial reporting. "
            "Known for careful documentation, organized follow-through, and clear communication with operations teams."
        ),
        "skills": [
            "General ledger", "Account reconciliation", "Journal entries", "Month-end close",
            "Financial reporting", "Accounts receivable", "Microsoft Excel", "GAAP"
        ],
        "experience": [
            {
                "title": "Staff Accountant", "company": "Cedar Works Manufacturing",
                "start": "2023-09", "end": "2026-08",
                "bullets": [
                    "Prepared journal entries and account reconciliations for monthly and year-end close activities.",
                    "Reviewed billing records and worked with operations staff to resolve transaction differences.",
                    "Produced recurring financial reports and documented supporting schedules for management review.",
                    "Supported external audit requests by organizing records and explaining account activity."
                ]
            },
            {
                "title": "Accounts Payable Specialist", "company": "Harbor Point Distribution",
                "start": "2021-09", "end": "2023-08",
                "bullets": [
                    "Reviewed invoices, matched purchasing records, and prepared approved payments.",
                    "Reconciled vendor statements and researched missing or duplicate transactions.",
                    "Maintained clear records and assisted with month-end account analysis."
                ]
            }
        ],
        "education": [{"credential": "Bachelor of Science in Accounting", "institution": "Pine Valley University", "year": 2021}],
        "certifications": [],
        "source_job_ids": ["3904098191", "3902768782", "3886886429"],
        "template": "compact"
    }
}

THEMES = {
    "classic": {"accent": "#17324D", "muted": "#5C6770", "heading": "Georgia", "body": "Arial", "rule": "#AAB7C4"},
    "modern": {"accent": "#176B68", "muted": "#4E6261", "heading": "Arial", "body": "Arial", "rule": "#B8D8D6"},
    "compact": {"accent": "#5B3A70", "muted": "#6B6270", "heading": "Georgia", "body": "Arial", "rule": "#D2C4DB"},
}


def month_index(value: str) -> int:
    year, month = map(int, value.split("-"))
    return year * 12 + month


def duration_months(profile: dict) -> int:
    return sum(month_index(job["end"]) - month_index(job["start"]) + 1 for job in profile["experience"])


def format_month(value: str) -> str:
    return datetime.strptime(value, "%Y-%m").strftime("%b %Y")


def resume_text(profile: dict) -> str:
    sections = [profile["headline"], "", "PROFESSIONAL SUMMARY", profile["summary"], "", "SKILLS", ", ".join(profile["skills"]), "", "EXPERIENCE"]
    for job in profile["experience"]:
        sections += [f"{job['title']} — {job['company']} | {format_month(job['start'])}–{format_month(job['end'])}"]
        sections += [f"• {bullet}" for bullet in job["bullets"]]
    sections += ["", "EDUCATION"]
    sections += [f"{item['credential']}, {item['institution']}, {item['year']}" for item in profile["education"]]
    if profile["certifications"]:
        sections += ["", "CERTIFICATIONS", *profile["certifications"]]
    return "\n".join(sections)


def resume_html(profile: dict, candidate_id: str) -> str:
    theme = THEMES[profile["template"]]
    esc = html.escape
    experience = "".join(
        f"<section><table class='job'><tr><td><b>{esc(job['title'])}</b></td>"
        f"<td class='dates'>{format_month(job['start'])} – {format_month(job['end'])}</td></tr></table>"
        f"<div class='company'>{esc(job['company'])}</div><ul>" + "".join(f"<li>{esc(b)}</li>" for b in job["bullets"]) + "</ul></section>"
        for job in profile["experience"]
    )
    education = "".join(f"<div><b>{esc(item['credential'])}</b><br>{esc(item['institution'])} · {item['year']}</div>" for item in profile["education"])
    certifications = ""
    if profile["certifications"]:
        certifications = "<h2>Certifications</h2><ul>" + "".join(f"<li>{esc(v)}</li>" for v in profile["certifications"]) + "</ul>"
    return f"""<html><head><style>
      * {{ box-sizing: border-box; }}
      body {{ font-family: {theme['body']}; color: #20262B; font-size: 9.4pt; line-height: 1.34; }}
      h1 {{ font-family: {theme['heading']}; color: {theme['accent']}; font-size: 22pt; margin: 5px 0 1px; }}
      .id {{ color: {theme['muted']}; font-size: 8pt; margin-bottom: 12px; }}
      h2 {{ font-family: {theme['heading']}; color: {theme['accent']}; font-size: 10pt; text-transform: uppercase;
            letter-spacing: 1px; border-bottom: 1px solid {theme['rule']}; padding-bottom: 3px; margin: 12px 0 6px; }}
      p {{ margin: 0 0 5px; }} ul {{ margin: 4px 0 6px 17px; padding: 0; }} li {{ margin-bottom: 2px; }}
      .skills {{ color: {theme['muted']}; }} .job {{ width: 100%; border-collapse: collapse; color: #1D252C; }}
      .job td {{ padding: 0; }} .job .dates {{ text-align: right; white-space: nowrap; font-size: 8.8pt; }}
      .company {{ color: {theme['muted']}; font-style: italic; margin-top: 1px; }}
    </style></head><body>
      <h1>{esc(profile['headline'])}</h1>
      <h2>Professional Summary</h2><p>{esc(profile['summary'])}</p>
      <h2>Skills</h2><p class='skills'>{esc('  ·  '.join(profile['skills']))}</p>
      <h2>Experience</h2>{experience}
      <h2>Education</h2>{education}{certifications}
    </body></html>"""


def write_pdf(profile: dict, candidate_id: str, path: Path) -> dict:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    result = page.insert_htmlbox(pymupdf.Rect(45, 25, 550, 795), resume_html(profile, candidate_id), scale_low=0.78)
    spare_height, scale = result
    if spare_height < 0:
        raise ValueError(f"Resume did not fit on one page: {candidate_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    document.set_metadata({"title": profile["headline"], "subject": "Resume"})
    document.save(path, garbage=4, deflate=True)
    document.close()
    return {"pages": 1, "render_scale": round(scale, 4), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def comparable_text(value: str) -> str:
    """Normalize PDF wrapping and common ligatures before checking visible text."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def recorded_path(path: Path) -> str:
    """Use project-relative paths for normal runs and absolute paths for custom outputs."""
    try:
        return str(path.relative_to(ACQUISITION_ROOT))
    except ValueError:
        return str(path.resolve())


def validate_profile(profile: dict) -> None:
    if not profile["skills"] or len(profile["skills"]) != len(set(profile["skills"])):
        raise ValueError("Skills must be present and unique")
    jobs = profile["experience"]
    if any(month_index(job["start"]) > month_index(job["end"]) for job in jobs):
        raise ValueError("Employment start date follows end date")
    for left, right in zip(jobs, jobs[1:]):
        if month_index(left["start"]) <= month_index(right["end"]):
            raise ValueError("Employment entries overlap or are out of newest-first order")
    months = duration_months(profile)
    if not profile["years_experience"] * 12 <= months < (profile["years_experience"] + 1) * 12:
        raise ValueError(f"Claimed experience does not match dates: {months} months")


def pilot_allocation(family_counts: dict[str, int], count: int) -> dict[str, int]:
    allocation = allocate_counts(family_counts, count)
    return {family: n for family, n in allocation.items() if n}


def generate(output_dir: Path, count: int = 3, seed: int = DEFAULT_SEED) -> dict:
    if count != 3:
        raise ValueError("This reviewed pilot supports exactly 3 profiles. Expand the profile generator before larger runs.")
    plan_path = authentic_data_directory() / "reports/role_families/family_counts.csv"
    plan = pd.read_csv(plan_path)
    counts = dict(zip(plan.role_family, plan.job_count, strict=True))
    allocation = pilot_allocation(counts, count)
    expected = set(PILOT_PROFILES)
    if set(allocation) != expected or any(value != 1 for value in allocation.values()):
        raise ValueError(f"Current ratios no longer select the reviewed pilot families: {allocation}")
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir, pdf_dir = output_dir / "text", output_dir / "pdf"
    text_dir.mkdir(exist_ok=True); pdf_dir.mkdir(exist_ok=True)
    records, pdf_records = [], []
    for number, family in enumerate(allocation, start=1):
        profile = json.loads(json.dumps(PILOT_PROFILES[family]))
        validate_profile(profile)
        candidate_id = f"syn_{GENERATOR_VERSION}_{number:04d}"
        text = resume_text(profile)
        text_path = text_dir / f"{candidate_id}.txt"
        text_path.write_text(text + "\n")
        pdf_path = pdf_dir / f"{candidate_id}.pdf"
        pdf_meta = write_pdf(profile, candidate_id, pdf_path)
        with pymupdf.open(pdf_path) as document:
            extracted = "\n".join(page.get_text() for page in document)
        required_phrases = [profile["headline"], *profile["skills"], *(job["title"] for job in profile["experience"])]
        extracted_comparable = comparable_text(extracted)
        missing = [phrase for phrase in required_phrases if comparable_text(phrase) not in extracted_comparable]
        if missing:
            raise ValueError(f"PDF text check failed for {candidate_id}: {missing}")
        record = {
            "candidate_id": candidate_id, "is_synthetic": True, "synthetic_stage": "resume_content",
            "generator_version": GENERATOR_VERSION, "seed": seed, "role_family": family,
            **profile, "calculated_experience_months": duration_months(profile),
            "resume_text": text, "resume_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_path": recorded_path(text_path), "pdf_path": recorded_path(pdf_path),
        }
        records.append(record); pdf_records.append({"candidate_id": candidate_id, **pdf_meta})
    profiles_path = output_dir / "profiles.jsonl"
    profiles_path.write_text("".join(json.dumps({k: v for k, v in record.items() if k != "resume_text"}) + "\n" for record in records))
    flat = pd.DataFrame([{**{k: v for k, v in r.items() if k not in {"experience", "education", "certifications", "skills", "source_job_ids"}},
                          "skills": r["skills"], "experience": json.dumps(r["experience"]),
                          "education": json.dumps(r["education"]), "certifications": r["certifications"],
                          "source_job_ids": r["source_job_ids"]} for r in records])
    flat.to_parquet(output_dir / "resumes.parquet", index=False)
    allocation_doc = {
        "requested_count": count, "method": "largest-remainder allocation over assigned authentic job-family counts",
        "full_allocation": allocate_counts(counts, count), "selected_allocation": allocation,
        "note": "With only three records, ratios select the three largest families. This is not a representative evaluation set."
    }
    (output_dir / "allocation.json").write_text(json.dumps(allocation_doc, indent=2) + "\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "generator_version": GENERATOR_VERSION,
        "seed": seed, "record_count": len(records), "synthetic_stage": "resume_content",
        "family_allocation": allocation, "profile_source": "reviewed fictional facts grounded in multiple authentic job descriptions",
        "generator_sha256": file_hash(Path(__file__)), "role_plan_sha256": file_hash(plan_path),
        "files": {"profiles": "profiles.jsonl", "table": "resumes.parquet", "text_directory": "text", "pdf_directory": "pdf"},
        "pdf_checks": pdf_records,
        "limitations": [
            "The three profiles are hand-authored pilot fixtures, not a scalable population generator.",
            "The role ratio determines families only; it does not validate skills or career histories.",
            "No names, contact details, relevance labels, outcomes, or latent marketplace variables are generated."
        ]
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=data_directory() / f"synthetic/generated/resumes/{GENERATOR_VERSION}")
    args = parser.parse_args()
    manifest = generate(args.output_dir, args.count, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
