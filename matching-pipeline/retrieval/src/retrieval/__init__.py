"""Leakage-safe building blocks for candidate retrieval."""

from .exact_search import exact_cosine_search
from .metrics import recall_at_k
from .renderers import render_candidate, render_job
from .schemas import CandidateProfile, JobProfile

__all__ = [
    "CandidateProfile",
    "JobProfile",
    "exact_cosine_search",
    "recall_at_k",
    "render_candidate",
    "render_job",
]
