"""Validated records shared by the synthetic-resume pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleTrack(StrictModel):
    track_name: str
    target_titles: list[str] = Field(min_length=3, max_length=8)
    core_skills: list[str] = Field(min_length=5, max_length=12)
    optional_skills: list[str] = Field(min_length=4, max_length=12)
    responsibilities: list[str] = Field(min_length=6, max_length=12)
    education_options: list[str] = Field(min_length=1, max_length=6)
    certifications: list[str] = Field(max_length=8)
    previous_titles: list[str] = Field(min_length=2, max_length=8)
    industries: list[str] = Field(min_length=3, max_length=8)

    @model_validator(mode="after")
    def unique_values(self):
        for field in (
            "target_titles", "core_skills", "optional_skills", "responsibilities",
            "education_options", "certifications", "previous_titles", "industries",
        ):
            values = getattr(self, field)
            if len(values) != len({value.casefold() for value in values}):
                raise ValueError(f"{field} contains duplicates")
        return self


class RoleArchetype(StrictModel):
    role_family: str
    role_type: str
    tracks: list[RoleTrack] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_tracks(self):
        names = [track.track_name.casefold() for track in self.tracks]
        if len(names) != len(set(names)):
            raise ValueError("track names contain duplicates")
        return self


class EducationSpec(StrictModel):
    credential: str
    institution: str
    graduation_year: int = Field(ge=1960, le=2026)


class ExperienceSpec(StrictModel):
    experience_id: str
    title: str
    company: str
    industry: str
    start: str
    end: str
    responsibilities: list[str] = Field(min_length=2, max_length=5)


class CandidateSpec(StrictModel):
    candidate_id: str
    generator_version: str
    seed: int
    role_family: str
    role_type: str
    role_track: str
    target_title: str
    seniority: Literal["entry", "mid", "senior", "leadership"]
    fit_profile: Literal["strong", "partial", "adjacent", "career_change"]
    years_experience: int = Field(ge=1, le=25)
    skills: list[str] = Field(min_length=4, max_length=14)
    experience: list[ExperienceSpec] = Field(min_length=1, max_length=4)
    education: list[EducationSpec] = Field(min_length=1, max_length=2)
    certifications: list[str] = Field(max_length=5)
    source_job_ids: list[str] = Field(min_length=1, max_length=12)


class ExperienceWriting(StrictModel):
    experience_id: str
    bullets: list[str] = Field(min_length=2, max_length=4)


class ResumeWriting(StrictModel):
    candidate_id: str
    professional_summary: str
    experience: list[ExperienceWriting]


def strict_schema(model: type[BaseModel]) -> dict:
    """Return the JSON Schema accepted by Responses API Structured Outputs."""
    return model.model_json_schema()
