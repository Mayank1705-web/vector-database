"""Candidate-count metrics for approximate vector-search evaluation."""

from __future__ import annotations


def candidate_fraction(
    candidate_count: int,
    total_vectors: int,
) -> float:
    """Calculate the fraction of vectors examined during a search."""
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise TypeError("candidate_count must be an integer.")

    if isinstance(total_vectors, bool) or not isinstance(total_vectors, int):
        raise TypeError("total_vectors must be an integer.")

    if candidate_count < 0:
        raise ValueError("candidate_count must not be negative.")

    if total_vectors <= 0:
        raise ValueError("total_vectors must be positive.")

    if candidate_count > total_vectors:
        raise ValueError(
            "candidate_count must not exceed total_vectors."
        )

    return candidate_count / total_vectors


__all__ = ["candidate_fraction"]