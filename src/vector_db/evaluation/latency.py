"""Latency statistics for vector-search evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics for a collection of query latencies."""

    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    mean: float


def calculate_percentiles(
    latencies: Sequence[Real],
) -> LatencyStats:
    """Calculate latency summary statistics.

    Parameters
    ----------
    latencies:
        Query latency measurements in seconds.

    Returns
    -------
    LatencyStats
        Minimum, maximum, mean, and p50/p95/p99 latency.

    Raises
    ------
    ValueError
        If the input is empty, contains negative values, or contains
        non-finite values.
    TypeError
        If a latency value is not numeric.
    """
    if not latencies:
        raise ValueError("Latency measurements must not be empty.")

    for latency in latencies:
        if isinstance(latency, bool) or not isinstance(latency, Real):
            raise TypeError("Latency measurements must be numeric.")

    values = np.asarray(latencies, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("Latency measurements must be finite.")

    if np.any(values < 0.0):
        raise ValueError("Latency measurements must not contain negative values.")

    return LatencyStats(
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
    )


__all__ = [
    "LatencyStats",
    "calculate_percentiles",
]