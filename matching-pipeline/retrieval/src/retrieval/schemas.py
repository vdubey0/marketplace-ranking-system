"""Typed, retrieval-time-safe views of candidate and job records."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _missing_to_none(value: Any) -> Any:
    """Convert scalar pandas missing values without inspecting list-like values."""
    if value is None:
        return None
    try:
        if value != value:  # NaN and pandas.NA scalars
            return None
    except (TypeError, ValueError):
        pass
    return value


def _text(value: Any) -> str | None:
    value = _missing_to_none(value)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


class _RetrievalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]):
        """Select only declared fields, preventing accidental label leakage."""
        return cls.model_validate({name: row.get(name) for name in cls.model_fields})


class CandidateProfile(_RetrievalRecord):
    candidate_id: str
    resume_text: str | None = None
    current_title: str | None = None
    skills: tuple[str, ...] = ()
    years_experience: float | None = Field(default=None, ge=0)
    education_level: str | None = None
    industries: tuple[str, ...] = ()
    domain: str | None = None
    role_family: str | None = None
    role_type: str | None = None
    role_track: str | None = None
    seniority: str | None = None
    certifications: tuple[str, ...] = ()

    @field_validator("candidate_id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        result = _text(value)
        if result is None:
            raise ValueError("candidate_id is required")
        return result

    @field_validator(
        "resume_text", "current_title", "education_level", "domain",
        "role_family", "role_type", "role_track", "seniority", mode="before"
    )
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        return _text(value)

    @field_validator("years_experience", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> Any:
        return _missing_to_none(value)

    @field_validator("skills", "industries", "certifications", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> tuple[str, ...]:
        value = _missing_to_none(value)
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        return tuple(text for item in value if (text := _text(item)) is not None)


class JobProfile(_RetrievalRecord):
    job_id: str
    title: str | None = None
    description: str | None = None
    role_family: str | None = None
    role_type: str | None = None
    role_track: str | None = None
    seniority: str | None = None
    minimum_years_experience: float | None = Field(default=None, ge=0)
    required_skills: tuple[str, ...] = ()
    preferred_skills: tuple[str, ...] = ()
    required_certifications: tuple[str, ...] = ()
    domain: str | None = None

    @field_validator("job_id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        result = _text(value)
        if result is None:
            raise ValueError("job_id is required")
        return result

    @field_validator(
        "title", "description", "role_family", "role_type", "role_track",
        "seniority", "domain", mode="before"
    )
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        return _text(value)

    @field_validator("minimum_years_experience", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> Any:
        return _missing_to_none(value)

    @field_validator(
        "required_skills", "preferred_skills", "required_certifications",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> tuple[str, ...]:
        value = _missing_to_none(value)
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        return tuple(text for item in value if (text := _text(item)) is not None)
