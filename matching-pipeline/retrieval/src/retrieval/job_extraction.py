"""Convert raw job descriptions into the structured retrieval schema."""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from .schemas import JobProfile


class JobExtraction(BaseModel):
    """All fields are required because the Responses API uses strict output."""
    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str | None
    role_family: str
    role_type: str
    role_track: str
    seniority: Literal["entry", "mid", "senior", "leadership"] | None
    minimum_years_experience: float | None
    required_skills: list[str]
    preferred_skills: list[str]
    required_certifications: list[str]
    domain: str | None


def extract_job_profiles(
    descriptions: Sequence[str],
    role_options: Sequence[dict],
    client: OpenAI,
    *,
    model: str,
) -> list[JobProfile]:
    """Extract one validated JobProfile per raw description."""
    if not descriptions:
        raise ValueError("provide at least one job description")
    profiles = []
    for number, description in enumerate(descriptions, start=1):
        job_id = f"input_job_{number:03d}"
        instructions = (
            "Extract the job into the supplied schema using only stated facts. "
            "Select exactly one supplied role option and copy its role_family, "
            "role_type, and role_track verbatim. Use null or an empty list when "
            "information is absent. Return the supplied job_id unchanged."
        )
        payload = {
            "job_id": job_id,
            "role_options": list(role_options),
            "job_description": str(description),
        }
        response = client.responses.create(
            model=model,
            store=False,
            reasoning={"effort": "low"},
            input=[
                {"role": "developer", "content": instructions},
                {"role": "user", "content": json.dumps(payload)},
            ],
            text={"format": {
                "type": "json_schema", "name": "job_profile", "strict": True,
                "schema": JobExtraction.model_json_schema(),
            }},
        )
        extracted = JobExtraction.model_validate_json(response.output_text)
        profile = JobProfile.model_validate({
            **extracted.model_dump(), "description": str(description),
        })
        if profile.job_id != job_id:
            raise ValueError(f"job ID changed: expected {job_id}, got {profile.job_id}")
        allowed = {(row["role_family"], row["role_type"], row["role_track"])
                   for row in role_options}
        if (profile.role_family, profile.role_type, profile.role_track) not in allowed:
            raise ValueError(f"invalid role selection for {job_id}")
        profiles.append(profile)
    return profiles
