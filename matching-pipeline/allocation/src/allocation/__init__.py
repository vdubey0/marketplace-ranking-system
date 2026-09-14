"""Global candidate-to-job allocation."""

from .optimizer import greedy_allocate, optimize_allocation

__all__ = ["greedy_allocate", "optimize_allocation"]
