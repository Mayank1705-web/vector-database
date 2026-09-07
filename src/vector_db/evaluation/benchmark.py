"""Benchmark exact and approximate vector search."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Sequence

import numpy as np
import numpy.typing as npt

from vector_db.approximate.ivf.index import IVFIndex
from vector_db.core.types import SearchResult
from vector_db.evaluation.candidates import candidate_fraction
from vector_db.evaluation.latency import LatencyStats, calculate_percentiles
from vector_db.evaluation.recall import recall_at_k
from vector_db.exact.brute_force import BruteForceIndex


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BenchmarkResult:
    """Structured results from an IVF-Flat benchmark."""

    n_vectors: int
    n_queries: int
    dimension: int

    n_clusters: int
    n_probe: int

    build_time: float

    recall: dict[int, float]
    latency: dict[int, LatencyStats]

    mean_candidate_count: float
    mean_candidate_fraction: float


def _validate_matrix(
    values: npt.ArrayLike,
    name: str,
) -> Array:
    """Convert and validate a 2-D finite matrix."""
    data = np.asarray(values, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")

    if data.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")

    if data.shape[1] == 0:
        raise ValueError(f"{name} must have a non-zero dimension.")

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return data


def _validate_positive_integer(
    value: int,
    name: str,
) -> int:
    """Validate a positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def _validate_k_values(
    k_values: Sequence[int],
    n_vectors: int,
) -> tuple[int, ...]:
    """Validate requested k values."""
    if not k_values:
        raise ValueError("k_values must not be empty.")

    validated: list[int] = []

    for k in k_values:
        k = _validate_positive_integer(k, "k")

        if k > n_vectors:
            raise ValueError(
                "k values must not exceed the number of vectors."
            )

        validated.append(k)

    if len(set(validated)) != len(validated):
        raise ValueError(
            "k_values must not contain duplicates."
        )

    return tuple(validated)


def _build_exact_index(
    vectors: Array,
) -> BruteForceIndex:
    """Build the exact brute-force correctness oracle."""
    index = BruteForceIndex()

    for vector_id, vector in enumerate(vectors):
        index.insert(vector_id, vector)

    return index


def _build_ivf_index(
    vectors: Array,
    n_clusters: int,
    n_probe: int,
    kmeans_iterations: int,
    seed: int,
) -> tuple[IVFIndex, float]:
    """Build an IVF-Flat index and measure its build time."""
    index = IVFIndex(
        n_clusters=n_clusters,
        n_probe=n_probe,
        kmeans_iterations=kmeans_iterations,
        seed=seed,
    )

    start = perf_counter()

    index.fit(vectors)

    build_time = perf_counter() - start

    return index, float(build_time)


def benchmark(
    vectors: npt.ArrayLike,
    queries: npt.ArrayLike,
    n_clusters: int,
    n_probe: int,
    k_values: Sequence[int] = (1, 5, 10),
    kmeans_iterations: int = 10,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark IVF-Flat against exact brute-force search.

    The brute-force index provides the ground-truth nearest neighbors.
    IVF-Flat is then evaluated for recall, query latency, and candidate
    reduction.

    Parameters
    ----------
    vectors:
        Database vectors with shape ``(n_vectors, dimension)``.

    queries:
        Query vectors with shape ``(n_queries, dimension)``.

    n_clusters:
        Number of IVF clusters.

    n_probe:
        Number of IVF clusters searched for each query.

    k_values:
        Values of k for which Recall@k and latency statistics are reported.

    kmeans_iterations:
        Number of K-Means iterations used during IVF training.

    seed:
        Random seed used by IVF training.

    Returns
    -------
    BenchmarkResult
        Structured benchmark measurements.

    Raises
    ------
    ValueError
        If input dimensions, sizes, or configuration values are invalid.
    TypeError
        If configuration values have invalid types.
    """
    data = _validate_matrix(vectors, "vectors")
    query_data = _validate_matrix(queries, "queries")

    if data.shape[1] != query_data.shape[1]:
        raise ValueError(
            "vectors and queries must have the same dimension."
        )

    n_vectors, dimension = data.shape
    n_queries = query_data.shape[0]

    n_clusters = _validate_positive_integer(
        n_clusters,
        "n_clusters",
    )

    n_probe = _validate_positive_integer(
        n_probe,
        "n_probe",
    )

    kmeans_iterations = _validate_positive_integer(
        kmeans_iterations,
        "kmeans_iterations",
    )

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")

    seed = int(seed)

    if n_clusters > n_vectors:
        raise ValueError(
            "n_clusters must not exceed the number of vectors."
        )

    if n_probe > n_clusters:
        raise ValueError(
            "n_probe must not exceed n_clusters."
        )

    validated_k_values = _validate_k_values(
        k_values,
        n_vectors,
    )

    # We only need to search up to the largest requested k.
    max_k = max(validated_k_values)

    # ---------------------------------------------------------------
    # 1. Build exact index.
    # ---------------------------------------------------------------
    exact_index = _build_exact_index(data)

    # ---------------------------------------------------------------
    # 2. Build IVF-Flat index.
    # ---------------------------------------------------------------
    ivf_index, build_time = _build_ivf_index(
        vectors=data,
        n_clusters=n_clusters,
        n_probe=n_probe,
        kmeans_iterations=kmeans_iterations,
        seed=seed,
    )

    if ivf_index.count != n_vectors:
        raise RuntimeError(
            "IVF index contains an unexpected number of vectors."
        )

    # ---------------------------------------------------------------
    # 3. Compute exact ground truth.
    # ---------------------------------------------------------------
    ground_truth: list[list[SearchResult]] = []

    for query in query_data:
        exact_results = exact_index.search(
            query,
            max_k,
        )

        ground_truth.append(exact_results)

    # ---------------------------------------------------------------
    # 4. Run IVF queries separately for each requested k.
    # ---------------------------------------------------------------
    approximate_results_by_k: dict[int, list[list[SearchResult]]] = {
        k: [] for k in validated_k_values
    }

    latencies_by_k: dict[int, list[float]] = {
        k: [] for k in validated_k_values
    }

    candidate_counts: list[int] = []

    for query in query_data:
        query_candidate_count: int | None = None

        for k in validated_k_values:
            start = perf_counter()

            results, candidate_count = ivf_index.search_with_stats(
                query,
                k,
            )

            elapsed = perf_counter() - start

            if not isinstance(elapsed, Real) or elapsed < 0:
                raise RuntimeError(
                    "Measured query latency must be non-negative."
                )

            approximate_results_by_k[k].append(results)
            latencies_by_k[k].append(float(elapsed))

            if query_candidate_count is None:
                query_candidate_count = int(candidate_count)
            elif int(candidate_count) != query_candidate_count:
                raise RuntimeError(
                    "Candidate count changed between k values."
                )

        if query_candidate_count is None:
            raise RuntimeError(
                "No candidate count was recorded for the query."
            )

        candidate_counts.append(query_candidate_count)

    # ---------------------------------------------------------------
    # 5. Calculate Recall@k.
    # ---------------------------------------------------------------
    recall: dict[int, float] = {}

    for k in validated_k_values:
        per_query_recall = [
            recall_at_k(
                true_results,
                approximate,
                k,
            )
            for true_results, approximate in zip(
                ground_truth,
                approximate_results_by_k[k],
            )
        ]

        recall[k] = float(
            np.mean(per_query_recall)
        )

    # ---------------------------------------------------------------
    # 6. Calculate independent latency statistics for each k.
    # ---------------------------------------------------------------
    latency: dict[int, LatencyStats] = {}

    for k in validated_k_values:
        latency[k] = calculate_percentiles(
            latencies_by_k[k]
        )

    # ---------------------------------------------------------------
    # 7. Calculate candidate statistics.
    # ---------------------------------------------------------------
    mean_candidate_count = float(
        np.mean(candidate_counts)
    )

    candidate_fractions = [
        candidate_fraction(
            count,
            n_vectors,
        )
        for count in candidate_counts
    ]

    mean_candidate_fraction = float(
        np.mean(candidate_fractions)
    )
    
    return BenchmarkResult(
        n_vectors=n_vectors,
        n_queries=n_queries,
        dimension=dimension,
        n_clusters=n_clusters,
        n_probe=n_probe,
        build_time=build_time,
        recall=recall,
        latency=latency,
        mean_candidate_count=mean_candidate_count,
        mean_candidate_fraction=mean_candidate_fraction,
    )


__all__ = [
    "BenchmarkResult",
    "benchmark",
]