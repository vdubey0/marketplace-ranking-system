"""Small, auditable exact cosine-search baseline."""
from __future__ import annotations

import numpy as np


def _normalized(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains a non-finite value")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero vector")
    return values / norms


def exact_cosine_search(
    job_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate row indices and scores, ordered deterministically."""
    if k <= 0:
        raise ValueError("k must be positive")
    jobs = _normalized(job_embeddings, "job_embeddings")
    candidates = _normalized(candidate_embeddings, "candidate_embeddings")
    if jobs.shape[1] != candidates.shape[1]:
        raise ValueError("job and candidate embedding dimensions must match")
    if candidates.shape[0] == 0:
        raise ValueError("candidate_embeddings cannot be empty")

    limit = min(k, candidates.shape[0])
    scores = jobs @ candidates.T
    indices = np.empty((jobs.shape[0], limit), dtype=np.int64)
    ordered_scores = np.empty((jobs.shape[0], limit), dtype=np.float32)
    candidate_rows = np.arange(candidates.shape[0])
    for row, similarities in enumerate(scores):
        # Candidate row is the secondary key, making equal-score ordering stable.
        order = np.lexsort((candidate_rows, -similarities))[:limit]
        indices[row] = order
        ordered_scores[row] = similarities[order]
    return indices, ordered_scores
