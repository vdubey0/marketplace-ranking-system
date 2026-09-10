"""Group existing job titles and propose synthetic-resume quantities. Generates no resumes."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ACQUISITION_ROOT))
from scripts.authentic.load_data import authentic_data_directory, file_hash


def normalize_title(title: str) -> str:
    """Keep role/seniority words; standardize punctuation for matching only."""
    text = unicodedata.normalize("NFKC", title).casefold()
    text = re.sub(r"[‐‑–—−_-]", " ", text)
    text = re.sub(r"\b(sr|jr)\.", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def compile_rules(config: dict) -> list:
    compiled = []
    for family in config["families"]:
        for role in family["roles"]:
            compiled.append((family["name"], role["name"], re.compile(role["pattern"]),
                             re.compile(role["exclude"]) if role.get("exclude") else None))
    return compiled


def classify_title(title: str, rules: list) -> dict:
    normalized = normalize_title(title)
    matches = []
    for family, role, pattern, exclude in rules:
        if pattern.search(normalized) and not (exclude and exclude.search(normalized)):
            matches.append((family, role))
    families = sorted({family for family, _ in matches})
    status = "assigned" if len(families) == 1 else "ambiguous" if families else "unmatched"
    return {
        "normalized_title": normalized,
        "assignment_status": status,
        "role_family": matches[0][0] if status == "assigned" else None,
        # Specific subroles precede generic roles within each family in the config.
        "role_type": matches[0][1] if status == "assigned" else None,
        "matched_families": " | ".join(families),
        "matched_role_types": " | ".join(dict.fromkeys(role for _, role in matches)),
    }


def allocate_counts(weights: dict[str, int], total: int) -> dict[str, int]:
    """Proportional whole-number quotas with deterministic largest-remainder rounding."""
    if total < 0:
        raise ValueError("Batch size cannot be negative")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Counts cannot be negative")
    denominator = sum(weights.values())
    if denominator == 0:
        if total:
            raise ValueError("Cannot allocate a positive batch without classified jobs")
        return {key: 0 for key in weights}
    # Integer arithmetic avoids rounding differences across platforms.
    allocation = {key: total * weight // denominator for key, weight in weights.items()}
    remainder = total - sum(allocation.values())
    order = sorted(weights, key=lambda k: (-(total * weights[k] % denominator), k))
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


def markdown_table(headers: list[str], rows: list[list]) -> str:
    def escape(value):
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(escape(v) for v in row) + " |" for row in rows),
    ])


def build_plan(data_dir: Path, rules_path: Path, batch_size: int) -> dict:
    if batch_size <= 0:
        raise ValueError("Choose a positive batch size")
    config = json.loads(rules_path.read_text())
    rules = compile_rules(config)
    jobs_path = data_dir / "processed/jobs.parquet"
    resumes_path = data_dir / "processed/resumes.parquet"
    jobs = pd.read_parquet(jobs_path, columns=["job_id", "title", "formatted_experience_level"])
    resumes = pd.read_parquet(resumes_path, columns=["resume_id", "category"])
    if jobs["job_id"].duplicated().any():
        raise ValueError("Expected deduplicated input jobs")
    titles = jobs["title"].fillna("").drop_duplicates().tolist()
    title_map = pd.DataFrame([{"title": title, **classify_title(title, rules)} for title in titles])
    jobs["title"] = jobs["title"].fillna("")
    mapped = jobs.merge(title_map, on="title", how="left", validate="many_to_one")
    assigned = mapped.loc[mapped.assignment_status.eq("assigned")]
    family_weights = assigned.role_family.value_counts().to_dict()
    quotas = allocate_counts(family_weights, batch_size)
    family_config = {f["name"]: f for f in config["families"]}
    category_counts = resumes.category.value_counts().to_dict()
    family_rows, role_rows = [], []
    for family, group in assigned.groupby("role_family", sort=True):
        examples = group.title.value_counts().head(3).index.tolist()
        categories = family_config[family]["resume_categories"]
        family_rows.append({
            "role_family": family,
            "job_count": len(group),
            "percent_of_all_jobs": round(100 * len(group) / len(jobs), 2),
            "percent_of_assigned_jobs": round(100 * len(group) / len(assigned), 2),
            "proposed_resumes": quotas[family],
            "example_titles": " | ".join(examples),
            "related_resume_categories": " | ".join(categories),
            # This is a coarse reference count; IT is deliberately shared, not divided.
            "related_category_resume_count": sum(category_counts.get(c, 0) for c in categories) if categories else None,
        })
        role_weights = group.role_type.value_counts().to_dict()
        role_quotas = allocate_counts(role_weights, quotas[family])
        for role, role_group in group.groupby("role_type", sort=True):
            role_rows.append({
                "role_family": family, "role_type": role, "job_count": len(role_group),
                "proposed_resumes": role_quotas[role],
                "example_titles": " | ".join(role_group.title.value_counts().head(3).index),
                "example_job_ids": " | ".join(role_group.sort_values("job_id").job_id.head(5).astype(str)),
            })
    families = pd.DataFrame(family_rows).sort_values(["job_count", "role_family"], ascending=[False, True])
    roles = pd.DataFrame(role_rows).sort_values(["role_family", "job_count"], ascending=[True, False])
    seniority = assigned.assign(source_experience_level=assigned.formatted_experience_level.fillna("Unknown"))
    seniority = seniority.groupby(["role_family", "source_experience_level"]).size().rename("job_count").reset_index()
    review = mapped.loc[~mapped.assignment_status.eq("assigned")].groupby(
        ["title", "assignment_status", "matched_families"], dropna=False
    ).agg(job_count=("job_id", "size"), example_job_id=("job_id", "first")).reset_index()
    review = review.sort_values(["job_count", "title"], ascending=[False, True])
    audit = pd.concat([
        group.sample(min(8, len(group)), random_state=42)
        for _, group in assigned.groupby("role_family")
    ], ignore_index=True)
    audit["reviewed_family"] = ""
    audit["review_notes"] = ""
    reports = data_dir / "reports/role_families"
    reports.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(data_dir / "processed/job_role_mapping.parquet", index=False)
    for name, frame in [("family_counts", families), ("resume_generation_plan", roles),
                        ("source_seniority_counts", seniority), ("titles_to_review", review),
                        ("assignment_audit_sample", audit)]:
        frame.to_csv(reports / f"{name}.csv", index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jobs_sha256": file_hash(jobs_path), "input_resumes_sha256": file_hash(resumes_path),
        "rules_sha256": file_hash(rules_path), "script_sha256": file_hash(Path(__file__)),
        "rule_version": config["version"], "input_jobs": len(jobs),
        "distinct_original_titles": len(titles), "status_counts": mapped.assignment_status.value_counts().to_dict(),
        "assigned_families": len(families), "role_types": len(roles), "proposed_batch_size": batch_size,
        "quota_method": "Proportional to assigned cleaned job counts; largest-remainder integer rounding. Ambiguous/unmatched excluded.",
        "synthetic_resumes_created": 0,
    }
    (reports / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    status = summary["status_counts"]
    rows = [[r.role_family, f"{r.job_count:,}", f"{r.percent_of_all_jobs:.2f}%", r.proposed_resumes, r.example_titles]
            for r in families.itertuples()]
    report = f"""# Which synthetic resumes should we create?

This is a draft generation plan based on our downloaded English job descriptions.
No synthetic resumes or PDFs have been created by this step.

## What was counted

- **{len(jobs):,}** cleaned job descriptions, with **{len(titles):,}** distinct original titles.
- **{len(assigned):,} ({len(assigned)/len(jobs):.1%})** jobs matched exactly one role family.
- **{status.get('ambiguous', 0):,}** jobs matched multiple families and need review.
- **{status.get('unmatched', 0):,}** jobs did not match a rule and need review.
- **{len(families)}** assigned role families, divided into **{len(roles)}** resume role types.

## Proposed first batch: {batch_size:,} resumes

Allocate resumes in proportion to the jobs assigned to each family. The quantities
sum to exactly {batch_size:,}. This is a starting mix for this dataset, not a claim
about the current labor market. The percentage column uses all {len(jobs):,} jobs;
the allocation uses only the {len(assigned):,} jobs assigned to one family.

{markdown_table(['Resume family', 'Jobs', 'Share of all jobs', 'Proposed resumes', 'Actual title examples'], rows)}

## How the grouping works

Rules match words in titles, not descriptions. Seniority words and original titles
are preserved. For example, “Software Engineer II” and “Senior Software Engineer”
belong to Software development. A generic software title does not tell us whether
someone should be a backend, frontend, or mobile developer; that specialty remains
unspecified unless the title says so. Description review is the next step.

Within a family, specific role rules run before generic ones. Across families,
multiple matches are flagged instead of resolved by guessing. Every job has its
original title, rule matches, and decision in `job_role_mapping.parquet`.

These are project-specific rules, not verified occupation labels. Do not use this
mapping as independent retrieval ground truth or treat it as a measured accuracy
result. Review the sample and unresolved titles before a large generation run.

## What the collected resumes contribute

The resume examples can guide wording and career-history structure. Their broad
source categories do not reliably identify each detailed role. The CSV includes
related source categories for reference, not confirmed matching candidates. IT
examples are referenced by more than one family, so those reference counts must
not be added together. No dedicated source category means unknown coverage, not
proof that no relevant resume exists.

## Experience levels

`source_seniority_counts.csv` preserves each family's supplied job experience levels,
including Unknown. These labels have not been converted to candidate years of
experience. Title words such as senior and manager can disagree with the supplied
labels; inspect the description before choosing candidate experience requirements.

## Files to inspect

- `data/authentic/processed/job_role_mapping.parquet`: one decision per input job.
- `data/authentic/reports/role_families/family_counts.csv`: family counts and proposed quantities.
- `data/authentic/reports/role_families/resume_generation_plan.csv`: finer role types, quantities,
  actual title examples, and job IDs to inspect before generating backgrounds.
- `data/authentic/reports/role_families/titles_to_review.csv`: ambiguous/unmatched titles, most frequent first.
- `data/authentic/reports/role_families/assignment_audit_sample.csv`: up to eight jobs per family
  for manual checking; reviewer fields start empty.
- `data/authentic/reports/role_families/manifest.json`: source hashes and reproducibility details.

Next: review uncertain titles, extract typical skills from descriptions for the
chosen roles, then create a small sample of fictional profiles and resumes. Use
multiple source jobs to understand a role; do not tailor each candidate to one job.
"""
    (ACQUISITION_ROOT / "docs").mkdir(exist_ok=True)
    (ACQUISITION_ROOT / "docs/resume_generation_plan.md").write_text(report)
    print(json.dumps(summary, indent=2))
    print(families[["role_family", "job_count", "proposed_resumes"]].to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1000, help="Proposed total resumes; nothing is generated")
    parser.add_argument("--rules", type=Path, default=ACQUISITION_ROOT / "config/role_families.json")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    build_plan(args.data_dir or authentic_data_directory(), args.rules, args.batch_size)


if __name__ == "__main__":
    main()
