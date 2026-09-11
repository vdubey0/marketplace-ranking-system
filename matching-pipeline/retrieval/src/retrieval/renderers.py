"""Deterministic text representations for the shared embedding space."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from .schemas import CandidateProfile, JobProfile

REPRESENTATION_VERSION = "v1"
UNKNOWN = "Unknown"


def _clean(value: str | None) -> str:
    if value is None:
        return UNKNOWN
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip() or UNKNOWN


def _items(values: Iterable[str]) -> str:
    # casefold is used only for ordering/deduplication; original spellings survive.
    unique = {re.sub(r"\s+", " ", unicodedata.normalize("NFKC", x)).strip() for x in values}
    return ", ".join(sorted(filter(None, unique), key=str.casefold)) or UNKNOWN


def _years(value: float | None) -> str:
    if value is None:
        return UNKNOWN
    return f"{value:g} years"


def _role(family: str | None, role_type: str | None, track: str | None) -> str:
    return " | ".join((_clean(family), _clean(role_type), _clean(track)))


def render_candidate(candidate: CandidateProfile) -> str:
    """Render only fields declared safe in CandidateProfile."""
    return "\n".join(
        (
            f"Current title: {_clean(candidate.current_title)}",
            f"Role: {_role(candidate.role_family, candidate.role_type, candidate.role_track)}",
            f"Seniority: {_clean(candidate.seniority)}",
            f"Experience: {_years(candidate.years_experience)}",
            f"Skills: {_items(candidate.skills)}",
            f"Industries: {_items(candidate.industries)}",
            f"Domain: {_clean(candidate.domain)}",
            f"Education: {_clean(candidate.education_level)}",
            f"Certifications: {_items(candidate.certifications)}",
            f"Resume: {_clean(candidate.resume_text)}",
        )
    )


def render_job(job: JobProfile) -> str:
    """Render only fields declared safe in JobProfile."""
    return "\n".join(
        (
            f"Title: {_clean(job.title)}",
            f"Role: {_role(job.role_family, job.role_type, job.role_track)}",
            f"Seniority: {_clean(job.seniority)}",
            f"Minimum experience: {_years(job.minimum_years_experience)}",
            f"Required skills: {_items(job.required_skills)}",
            f"Preferred skills: {_items(job.preferred_skills)}",
            f"Domain: {_clean(job.domain)}",
            f"Required certifications: {_items(job.required_certifications)}",
            f"Description: {_clean(job.description)}",
        )
    )
