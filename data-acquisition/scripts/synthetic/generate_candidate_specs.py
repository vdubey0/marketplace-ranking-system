"""Create deterministic, internally consistent candidate specifications."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.authentic.load_data import authentic_data_directory, data_directory, file_hash
from scripts.planning.plan_resumes import allocate_counts
from scripts.synthetic.io_utils import read_jsonl, write_jsonl
from scripts.synthetic.models import CandidateSpec, EducationSpec, ExperienceSpec, RoleArchetype

VERSION = "v1"
LOG = logging.getLogger(__name__)
DEFAULT_CATALOG = data_directory() / "synthetic/generated/role_catalog/v1/role_archetypes.jsonl"
DEFAULT_OUTPUT = data_directory() / f"synthetic/generated/candidate_specs/{VERSION}"

COMPANY_PREFIXES = ["Northfield", "Cedar Ridge", "Blue Harbor", "Westbridge", "Juniper", "Redwood", "Clearwater", "Stonegate", "Lakeview", "Brightpath", "Silver Oak", "Greenline"]
COMPANY_SUFFIXES = ["Services", "Partners", "Group", "Solutions", "Industries", "Associates", "Systems", "Collective"]
INSTITUTIONS = ["Lakeshore State University", "Westbridge College", "Pine Valley University", "North Coast Institute", "Cedar Grove College", "Franklin Metropolitan University"]
FIT_WEIGHTS = {"strong": 50, "partial": 30, "adjacent": 15, "career_change": 5}
LEVEL_YEARS = {"entry": (1, 2), "mid": (3, 6), "senior": (7, 12), "leadership": (10, 18)}
LEVEL_MAP = {
    "Internship": "entry", "Entry level": "entry", "Associate": "mid",
    "Mid-Senior level": "senior", "Director": "leadership", "Executive": "leadership",
}


def shift_month(value: str, delta: int) -> str:
    year, month = map(int, value.split("-"))
    index = year * 12 + month - 1 + delta
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def choose_seniority(rng: random.Random, family: str, seniority: pd.DataFrame) -> str:
    group = seniority.loc[seniority.role_family.eq(family)]
    weights = Counter()
    for row in group.itertuples(index=False):
        mapped = LEVEL_MAP.get(str(row.source_experience_level))
        if mapped:
            weights[mapped] += int(row.job_count)
    if not weights:
        weights.update({"entry": 20, "mid": 45, "senior": 30, "leadership": 5})
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def title_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold())) - {"senior", "sr", "junior", "jr", "ii", "iii"}


def track_weights(archetype: RoleArchetype, titles: pd.Series) -> list[int]:
    """Estimate within-role track prevalence from authentic title frequencies."""
    targets = [[title_tokens(title) for title in track.target_titles] for track in archetype.tracks]
    weights = [0] * len(archetype.tracks)
    for title, count in titles.value_counts().items():
        tokens = title_tokens(str(title))
        scores = [max((len(tokens & option) / len(tokens | option) for option in options), default=0) for options in targets]
        best = max(scores)
        if best >= 0.5 and scores.count(best) == 1:
            weights[scores.index(best)] += int(count)
    return weights if sum(weights) else [1] * len(archetype.tracks)


def distribute_months(total: int, jobs: int, rng: random.Random) -> list[int]:
    if jobs == 1:
        return [total]
    cuts = sorted(rng.sample(range(12, total - 11), jobs - 1))
    values = [cuts[0], *(b - a for a, b in zip(cuts, cuts[1:])), total - cuts[-1]]
    if min(values) < 10:
        return distribute_months(total, jobs, rng)
    return values


def make_candidate(number: int, archetype_record: dict, seniority_data: pd.DataFrame,
                   role_track_weights: list[int], rng: random.Random, seed: int) -> CandidateSpec:
    source_job_ids = [str(value) for value in archetype_record.pop("source_job_ids")]
    archetype = RoleArchetype.model_validate(archetype_record)
    track = rng.choices(archetype.tracks, weights=role_track_weights, k=1)[0]
    level = choose_seniority(rng, archetype.role_family, seniority_data)
    years = rng.randint(*LEVEL_YEARS[level])
    fit = rng.choices(list(FIT_WEIGHTS), weights=list(FIT_WEIGHTS.values()), k=1)[0]
    core_fraction = {"strong": .9, "partial": .7, "adjacent": .55, "career_change": .45}[fit]
    core_count = max(4, round(len(track.core_skills) * core_fraction))
    optional_count = {"strong": 3, "partial": 2, "adjacent": 1, "career_change": 0}[fit]
    skills = rng.sample(track.core_skills, min(core_count, len(track.core_skills)))
    skills += rng.sample(track.optional_skills, min(optional_count, len(track.optional_skills)))
    rng.shuffle(skills)
    job_count = 1 if years <= 2 else 2 if years <= 7 else 3
    total_months = years * 12 + rng.randint(0, 8)
    durations = distribute_months(total_months, job_count, rng)
    end = "2026-08"
    experiences = []
    target_title = rng.choice(track.target_titles)
    earlier_titles = rng.sample(track.previous_titles, k=min(job_count - 1, len(track.previous_titles)))
    titles = [target_title, *earlier_titles]
    for index, (title, months) in enumerate(zip(titles, reversed(durations), strict=True), start=1):
        start = shift_month(end, -(months - 1))
        responsibilities = rng.sample(track.responsibilities, k=min(4 if index == 1 else 3, len(track.responsibilities)))
        experiences.append(ExperienceSpec(
            experience_id=f"exp-{index}", title=title,
            company=f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)}",
            industry=rng.choice(track.industries), start=start, end=end,
            responsibilities=responsibilities,
        ))
        end = shift_month(start, -rng.randint(1, 3))
    earliest_year = int(experiences[-1].start[:4])
    education = EducationSpec(
        credential=rng.choice(track.education_options),
        institution=rng.choice(INSTITUTIONS),
        graduation_year=min(earliest_year, 2026),
    )
    certifications = rng.sample(
        track.certifications,
        k=min(len(track.certifications), rng.choice([0, 0, 1, 1, 2])),
    )
    return CandidateSpec(
        candidate_id=f"syn_{VERSION}_{number:06d}", generator_version=VERSION, seed=seed,
        role_family=archetype.role_family, role_type=archetype.role_type, role_track=track.track_name,
        target_title=target_title, seniority=level, fit_profile=fit,
        years_experience=years, skills=skills, experience=experiences,
        education=[education], certifications=certifications,
        source_job_ids=source_job_ids,
    )


def generate(catalog_path: Path, output_dir: Path, count: int, seed: int) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    records = read_jsonl(catalog_path)
    if not records:
        raise ValueError("role catalog is empty")
    keys = [(r["role_family"], r["role_type"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("role catalog contains duplicate role types")
    authentic = authentic_data_directory()
    plan = pd.read_csv(authentic / "reports/role_families/resume_generation_plan.csv")
    seniority = pd.read_csv(authentic / "reports/role_families/source_seniority_counts.csv")
    mapping = pd.read_parquet(
        authentic / "processed/job_role_mapping.parquet",
        columns=["title", "role_family", "role_type"],
    )
    weights = {(r.role_family, r.role_type): int(r.job_count) for r in plan.itertuples(index=False) if (r.role_family, r.role_type) in set(keys)}
    allocation = allocate_counts({f"{family}\t{role}": weight for (family, role), weight in weights.items()}, count)
    by_key = {(r["role_family"], r["role_type"]): r for r in records}
    rng = random.Random(seed)
    candidates, number = [], 1
    progress = tqdm(total=count, desc="Candidate specifications", unit="resume", dynamic_ncols=True)
    for joined_key, quantity in allocation.items():
        family, role = joined_key.split("\t", 1)
        titles = mapping.loc[mapping.role_family.eq(family) & mapping.role_type.eq(role), "title"]
        archetype_data = dict(by_key[(family, role)])
        archetype_for_weights = dict(archetype_data)
        archetype_for_weights.pop("source_job_ids")
        weights_for_tracks = track_weights(RoleArchetype.model_validate(archetype_for_weights), titles)
        for _ in range(quantity):
            candidates.append(make_candidate(number, dict(archetype_data), seniority, weights_for_tracks, rng, seed))
            number += 1
            progress.update(1)
    progress.close()
    rng.shuffle(candidates)
    # IDs stay unique but are reassigned after shuffling so file order has no family blocks.
    final = []
    for number, candidate in enumerate(candidates, start=1):
        final.append(candidate.model_copy(update={"candidate_id": f"syn_{VERSION}_{number:06d}"}))
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_path = output_dir / "candidate_specs.jsonl"
    write_jsonl(specs_path, (item.model_dump() for item in final))
    allocation_rows = [{"role_family": key.split("\t", 1)[0], "role_type": key.split("\t", 1)[1], "count": value}
                       for key, value in allocation.items()]
    (output_dir / "allocation.json").write_text(json.dumps(allocation_rows, indent=2) + "\n")
    manifest = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "record_count": len(final), "role_count": len(records),
        "catalog_sha256": file_hash(catalog_path), "specs_sha256": file_hash(specs_path),
        "fit_profile_counts": Counter(item.fit_profile for item in final),
        "seniority_counts": Counter(item.seniority for item in final),
        "role_track_counts": Counter(item.role_track for item in final),
    }
    manifest["fit_profile_counts"] = dict(manifest["fit_profile_counts"])
    manifest["seniority_counts"] = dict(manifest["seniority_counts"])
    manifest["role_track_counts"] = dict(manifest["role_track_counts"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("Generating %s candidate specifications with seed %s", args.count, args.seed)
    print(json.dumps(generate(args.catalog, args.output_dir, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()
