"""Recall metrics for evaluating approximate vector search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from vector_db.core.types import SearchResult


T = TypeVar("T")


def _extract_id(result: int | SearchResult) -> int:
    """Extract an ID from either an integer ID or SearchResult."""
    if isinstance(result, SearchResult):
        return result.id

    if isinstance(result, bool) or not isinstance(result, int):
        raise TypeError("Results must contain integer IDs or SearchResult objects.")

    return result


def recall_at_k(
    ground_truth: Sequence[int | SearchResult],
    approximate_results: Sequence[int | SearchResult],
    k: int,
) -> float:
    """Calculate Recall@k for one query.

    Recall@k is the fraction of the true top-k IDs that are
    also present in the approximate top-k results.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be positive.")

    if not ground_truth or not approximate_results:
        return 0.0

    true_ids = {
        _extract_id(result)
        for result in ground_truth[:k]
    }

    approximate_ids = {
        _extract_id(result)
        for result in approximate_results[:k]
    }

    if not true_ids:
        return 0.0

    return len(true_ids & approximate_ids) / len(true_ids)


def mean_recall_at_k(
    ground_truth_results: Sequence[Sequence[int | SearchResult]],
    approximate_results: Sequence[Sequence[int | SearchResult]],
    k: int,
) -> float:
    """Calculate mean Recall@k across multiple queries."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be positive.")

    if len(ground_truth_results) != len(approximate_results):
        raise ValueError(
            "Ground-truth and approximate result counts must match."
        )

    if not ground_truth_results:
        return 0.0

    recalls = [
        recall_at_k(
            ground_truth,
            approximate,
            k,
        )
        for ground_truth, approximate in zip(
            ground_truth_results,
            approximate_results,
        )
    ]

    return sum(recalls) / len(recalls)


__all__ = [
    "mean_recall_at_k",
    "recall_at_k",
]