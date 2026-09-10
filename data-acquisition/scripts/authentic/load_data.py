"""Download and load public English resume examples and job postings. No EDA.

Run: .venv/bin/python data-acquisition/scripts/authentic/load_data.py
Cached archives are reused. --offline forbids downloads; --refresh redownloads.
The resume examples are not independently verified real-person records.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import pandas as pd
import requests
from dotenv import load_dotenv
from lingua import LanguageDetectorBuilder
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ACQUISITION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ACQUISITION_ROOT.parent
SOURCES = {
    "resumes": {
        "slug": "snehaanbhawal/resume-dataset",
        "archive": "resume-dataset.zip",
        "member": "Resume/Resume.csv",
        "text_column": "resume_text",
        "id_column": "resume_id",
        "origin": "published_resume_example",
    },
    "jobs": {
        "slug": "arshkon/linkedin-job-postings",
        "archive": "linkedin-job-postings.zip",
        "member": "postings.csv",
        "text_column": "description",
        "id_column": "job_id",
        "origin": "public_job_posting",
    },
}
LOG = logging.getLogger(__name__)


def data_directory() -> Path:
    """Return the shared data-acquisition data directory."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(os.getenv("DATA_DIR", str(ACQUISITION_ROOT / "data"))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def authentic_data_directory() -> Path:
    """Return the raw, processed, and report directory for authentic sources."""
    return data_directory() / "authentic"


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_archive(source: dict, destination: Path, *, offline=False, refresh=False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not refresh:
        if not is_zipfile(destination):
            raise ValueError(f"Invalid ZIP: {destination}. Run again with --refresh.")
        LOG.info("Using cached %s", destination.name)
        return destination
    if offline:
        raise FileNotFoundError(f"Offline mode: archive not found at {destination}")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{source['slug']}"
    temporary = destination.with_suffix(".zip.part")
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        LOG.info("Downloading %s", source["slug"])
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
    if not is_zipfile(temporary):
        raise ValueError(f"Expected ZIP, got another response at {temporary}. Check Kaggle access.")
    with ZipFile(temporary) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")
    temporary.replace(destination)
    return destination


def read_csv_member(archive_path: Path, member: str) -> pd.DataFrame:
    with ZipFile(archive_path) as archive, archive.open(member) as handle:
        return pd.read_csv(handle, dtype="string", encoding="utf-8-sig")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()


def read_source(kind: str, archive_path: Path) -> pd.DataFrame:
    """Preserve original source fields while adding explicit source identity."""
    source = SOURCES[kind]
    frame = read_csv_member(archive_path, source["member"])
    if kind == "resumes":
        frame = frame.rename(columns={"ID": "resume_id", "Resume_str": "resume_text", "Category": "category"})
        # Original HTML and PDFs remain available in the immutable archive.
        frame = frame.drop(columns=["Resume_html"], errors="ignore")
        with ZipFile(archive_path) as archive:
            pdfs = {Path(name).stem: name for name in archive.namelist() if name.endswith(".pdf")}
        frame["pdf_archive_member"] = frame["resume_id"].map(pdfs).astype("string")
        frame["authenticity_status"] = "real_person_not_independently_verified"
    else:
        tags = read_csv_member(archive_path, "jobs/job_skills.csv")
        mapping = read_csv_member(archive_path, "mappings/skills.csv")
        tags = tags.merge(mapping, on="skill_abr", how="left", validate="many_to_one")
        names = tags.groupby("job_id")["skill_name"].agg(
            lambda values: " | ".join(sorted(set(values.dropna())))
        )
        # These are source job-function tags (e.g. Engineering), not extracted technical skills.
        frame["source_skill_tags"] = frame["job_id"].map(names)
        for column in ["listed_time", "original_listed_time", "expiry", "closed_time"]:
            frame[column + "_utc"] = pd.to_datetime(
                pd.to_numeric(frame[column], errors="coerce"), unit="ms", utc=True, errors="coerce"
            )
    frame["source_row"] = range(len(frame))
    frame["source"] = source["slug"]
    frame["source_url"] = "https://www.kaggle.com/datasets/" + source["slug"]
    frame["content_origin"] = source["origin"]
    return frame


def classify_texts(texts: list[str]) -> list[tuple[str, float]]:
    """Detect dominant language on up to 2,000 characters, sampled across each text.

    Lingua scores are relative confidence, not calibrated probabilities. This is
    a practical English filter, not a guarantee that every sentence is English.
    """
    # Default accuracy avoids false rejections triggered by unusual product names
    # in low-accuracy mode (e.g. an English HR resume mentioning "OrgPlus").
    detector = LanguageDetectorBuilder.from_all_languages().build()
    samples = []
    for text in texts:
        if len(text) > 2000:
            middle = len(text) // 2
            text = text[:700] + " " + text[middle:middle + 600] + " " + text[-700:]
        samples.append(text)
    result = []
    for start in range(0, len(samples), 1000):
        batch = samples[start:start + 1000]
        predictions = detector.compute_language_confidence_values_in_parallel(batch)
        for text, scores in zip(batch, predictions, strict=True):
            if not text or not scores or scores[0].value == 0:
                result.append(("unknown", 0.0))
            else:
                best = scores[0]
                result.append((best.language.iso_code_639_1.name.lower(), best.value))
        if (start + len(batch)) % 10000 == 0:
            LOG.info("Language checked: %s / %s unique texts", start + len(batch), len(texts))
    return result


def clean_source(frame: pd.DataFrame, kind: str, min_confidence: float):
    """Return accepted English rows, a per-row audit, and loading counts."""
    source = SOURCES[kind]
    frame = frame.copy()
    column = source["text_column"]
    identity = source["id_column"]
    frame[column] = frame[column].map(normalize_text)
    frame["text_sha256"] = frame[column].map(lambda x: hashlib.sha256(x.casefold().encode()).hexdigest())
    unique = frame.drop_duplicates("text_sha256")
    LOG.info("Checking language of %s %s texts", len(unique), kind)
    languages = pd.DataFrame(classify_texts(unique[column].tolist()), columns=["language", "language_confidence"])
    languages["text_sha256"] = unique["text_sha256"].to_numpy()
    frame = frame.merge(languages, on="text_sha256", how="left", validate="many_to_one")
    frame["status"] = "accepted"
    frame.loc[frame[identity].isna() | frame[identity].fillna("").str.strip().eq(""), "status"] = "missing_id"
    frame.loc[frame[column].eq(""), "status"] = "empty_text"
    eligible = frame["status"].eq("accepted")
    frame.loc[eligible & frame["language"].ne("en"), "status"] = "non_english_or_unknown"
    frame.loc[eligible & frame["language"].eq("en") & frame["language_confidence"].lt(min_confidence), "status"] = "uncertain_english"
    eligible_rows = frame.loc[frame["status"].eq("accepted")]
    duplicates = eligible_rows.duplicated(identity) | eligible_rows.duplicated("text_sha256")
    frame.loc[eligible_rows.index[duplicates], "status"] = "duplicate_id_or_text"
    # Same text under different source IDs is grouped. All original rows remain in raw ZIPs.
    audit = frame[[identity, "source_row", "text_sha256", "language", "language_confidence", "status"]].copy()
    counts = {str(k): int(v) for k, v in frame["status"].value_counts().items()}
    counts["raw_rows"] = len(frame)
    accepted = frame.loc[frame["status"].eq("accepted")].drop(columns="status").reset_index(drop=True)
    return accepted, audit, counts


def load_datasets(data_dir: Path | None = None, *, offline=False, refresh=False, min_confidence=None) -> dict:
    """Load both sources to Parquet and write a reproducibility manifest."""
    default_dir = authentic_data_directory()
    data_dir = Path(data_dir) if data_dir is not None else default_dir
    threshold = float(os.getenv("ENGLISH_MIN_CONFIDENCE", "0.70")) if min_confidence is None else min_confidence
    if not 0 <= threshold <= 1:
        raise ValueError("English confidence must be between 0 and 1")
    if offline and refresh:
        raise ValueError("--offline and --refresh cannot be combined")
    output = data_dir / "processed"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "english_min_confidence": threshold,
        "language_method": "lingua_all_languages_default_accuracy_2000_chars_start_middle_end",
        "loader_sha256": file_hash(Path(__file__)),
        "sources": {},
    }
    for kind, source in SOURCES.items():
        archive = download_archive(source, data_dir / "raw" / kind / source["archive"], offline=offline, refresh=refresh)
        frame, audit, counts = clean_source(read_source(kind, archive), kind, threshold)
        for table, filename in [(frame, f"{kind}.parquet"), (audit, f"{kind}_audit.parquet")]:
            path = output / filename
            temporary = path.with_suffix(".parquet.part")
            table.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        manifest["sources"][kind] = {
            **source, "url": "https://www.kaggle.com/datasets/" + source["slug"],
            "archive_sha256": file_hash(archive), "archive_bytes": archive.stat().st_size,
            "counts": counts, "output": f"{kind}.parquet",
        }
        LOG.info("%s: %s", kind, counts)
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use existing ZIPs; never access the network")
    parser.add_argument("--refresh", action="store_true", help="Download new source ZIPs, replacing cached archives")
    parser.add_argument("--data-dir", type=Path, help="Override DATA_DIR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_datasets(args.data_dir, offline=args.offline, refresh=args.refresh)


if __name__ == "__main__":
    main()
