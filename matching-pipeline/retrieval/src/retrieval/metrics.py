"""Retrieval metrics with explicit edge-case behavior."""
from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float | None:
    """Return recall for one job, or None when that job has no positives."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return None
    # Repeated IDs must not receive repeated credit.
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def recalls_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    ks: Iterable[int],
) -> dict[int, float | None]:
    return {k: recall_at_k(retrieved_ids, relevant_ids, k) for k in ks}
