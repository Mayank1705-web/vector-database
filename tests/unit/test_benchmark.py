"""Tests for the vector-search benchmark orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from vector_db.evaluation.benchmark import (
    BenchmarkResult,
    benchmark,
)


def make_vectors() -> np.ndarray:
    """Create a small deterministic database dataset."""
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
            [-1.0, 0.0],
            [-0.9, -0.1],
            [0.0, -1.0],
            [0.1, -0.9],
        ],
        dtype=np.float64,
    )


def make_queries() -> np.ndarray:
    """Create deterministic queries with known nearest vectors."""
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_benchmark_returns_structured_result() -> None:
    """Benchmark returns a BenchmarkResult with the expected fields."""
    result = benchmark(
        vectors=make_vectors(),
        queries=make_queries(),
        n_clusters=2,
        n_probe=2,
        k_values=(1, 5),
        kmeans_iterations=5,
        seed=42,
    )

    assert isinstance(result, BenchmarkResult)

    assert result.n_vectors == 10
    assert result.n_queries == 3
    assert result.dimension == 2

    assert result.n_clusters == 2
    assert result.n_probe == 2

    assert result.build_time >= 0.0

    assert set(result.recall) == {1, 5}
    assert set(result.latency) == {1, 5}

    assert result.mean_candidate_count >= 0.0
    assert 0.0 <= result.mean_candidate_fraction <= 1.0


def test_benchmark_recall_is_perfect_when_all_clusters_are_probed() -> None:
    """Probing every cluster should reproduce exact search results."""
    result = benchmark(
        vectors=make_vectors(),
        queries=make_queries(),
        n_clusters=2,
        n_probe=2,
        k_values=(1, 5),
        kmeans_iterations=5,
        seed=42,
    )

    assert result.recall[1] == pytest.approx(1.0)
    assert result.recall[5] == pytest.approx(1.0)


def test_benchmark_latency_contains_percentile_statistics() -> None:
    """Each requested k has complete latency statistics."""
    result = benchmark(
        vectors=make_vectors(),
        queries=make_queries(),
        n_clusters=2,
        n_probe=1,
        k_values=(1, 5),
        kmeans_iterations=5,
        seed=42,
    )

    for k in (1, 5):
        stats = result.latency[k]

        assert stats.p50 >= 0.0
        assert stats.p95 >= stats.p50
        assert stats.p99 >= stats.p95
        assert stats.minimum >= 0.0
        assert stats.maximum >= stats.minimum
        assert stats.mean >= 0.0


def test_benchmark_candidate_metrics_are_valid() -> None:
    """Candidate statistics stay within valid bounds."""
    result = benchmark(
        vectors=make_vectors(),
        queries=make_queries(),
        n_clusters=2,
        n_probe=1,
        k_values=(1,),
        kmeans_iterations=5,
        seed=42,
    )

    assert 0.0 < result.mean_candidate_count <= result.n_vectors
    assert 0.0 < result.mean_candidate_fraction <= 1.0


def test_benchmark_candidate_fraction_matches_count() -> None:
    """Candidate fraction equals candidate count divided by database size."""
    result = benchmark(
        vectors=make_vectors(),
        queries=make_queries(),
        n_clusters=2,
        n_probe=1,
        k_values=(1,),
        kmeans_iterations=5,
        seed=42,
    )

    expected_fraction = result.mean_candidate_count / result.n_vectors

    assert result.mean_candidate_fraction == pytest.approx(
        expected_fraction
    )


def test_benchmark_supports_multiple_k_values() -> None:
    """Benchmark produces independent recall and latency results for each k."""
    result = benchmark(
        vectors=make_vectors(),
        queries=make_queries(),
        n_clusters=2,
        n_probe=2,
        k_values=(1, 3, 5),
        kmeans_iterations=5,
        seed=42,
    )

    assert list(result.recall.keys()) == [1, 3, 5]
    assert list(result.latency.keys()) == [1, 3, 5]

    for k in (1, 3, 5):
        assert 0.0 <= result.recall[k] <= 1.0


def test_benchmark_rejects_empty_vectors() -> None:
    """Benchmark rejects an empty database."""
    vectors = np.empty((0, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="vectors"):
        benchmark(
            vectors=vectors,
            queries=make_queries(),
            n_clusters=2,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_empty_queries() -> None:
    """Benchmark rejects an empty query set."""
    queries = np.empty((0, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="queries"):
        benchmark(
            vectors=make_vectors(),
            queries=queries,
            n_clusters=2,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_vector_dimension_mismatch() -> None:
    """Benchmark rejects vectors and queries with different dimensions."""
    queries = np.array(
        [
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="dimension"):
        benchmark(
            vectors=make_vectors(),
            queries=queries,
            n_clusters=2,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_invalid_k() -> None:
    """Benchmark rejects non-positive k values."""
    with pytest.raises(ValueError, match="positive"):
        benchmark(
            vectors=make_vectors(),
            queries=make_queries(),
            n_clusters=2,
            n_probe=1,
            k_values=(0,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_k_larger_than_database() -> None:
    """Benchmark rejects k values larger than the database."""
    with pytest.raises(ValueError, match="vectors"):
        benchmark(
            vectors=make_vectors(),
            queries=make_queries(),
            n_clusters=2,
            n_probe=1,
            k_values=(11,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_invalid_n_probe() -> None:
    """Benchmark rejects n_probe greater than n_clusters."""
    with pytest.raises(ValueError):
        benchmark(
            vectors=make_vectors(),
            queries=make_queries(),
            n_clusters=2,
            n_probe=3,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_too_many_clusters() -> None:
    """Benchmark rejects more clusters than vectors."""
    with pytest.raises(ValueError):
        benchmark(
            vectors=make_vectors(),
            queries=make_queries(),
            n_clusters=11,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_non_2d_vectors() -> None:
    """Benchmark requires a 2-D vector matrix."""
    vectors = np.ones(20, dtype=np.float64)

    with pytest.raises(ValueError, match="2-D"):
        benchmark(
            vectors=vectors,
            queries=make_queries(),
            n_clusters=2,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_non_2d_queries() -> None:
    """Benchmark requires a 2-D query matrix."""
    queries = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match="2-D"):
        benchmark(
            vectors=make_vectors(),
            queries=queries,
            n_clusters=2,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_non_finite_vectors() -> None:
    """Benchmark rejects NaN or infinite vectors."""
    vectors = make_vectors()
    vectors[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        benchmark(
            vectors=vectors,
            queries=make_queries(),
            n_clusters=2,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_rejects_non_finite_queries() -> None:
    """Benchmark rejects NaN or infinite queries."""
    queries = make_queries()
    queries[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        benchmark(
            vectors=make_vectors(),
            queries=queries,
            n_clusters=2,
            n_probe=1,
            k_values=(1,),
            kmeans_iterations=5,
            seed=42,
        )


def test_benchmark_is_reproducible_for_recall_and_candidates() -> None:
    """Same data and configuration produce the same quality metrics."""
    vectors = make_vectors()
    queries = make_queries()

    result_1 = benchmark(
        vectors=vectors,
        queries=queries,
        n_clusters=2,
        n_probe=1,
        k_values=(1, 5),
        kmeans_iterations=5,
        seed=42,
    )

    result_2 = benchmark(
        vectors=vectors,
        queries=queries,
        n_clusters=2,
        n_probe=1,
        k_values=(1, 5),
        kmeans_iterations=5,
        seed=42,
    )

    assert result_1.recall == result_2.recall
    assert result_1.mean_candidate_count == pytest.approx(
        result_2.mean_candidate_count
    )
    assert result_1.mean_candidate_fraction == pytest.approx(
        result_2.mean_candidate_fraction
    )
def test_benchmark_measures_latency_separately_for_each_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each requested k should receive its own latency measurements."""
    vectors = make_vectors()
    queries = make_queries()

    recorded_k_values: list[int] = []

    from vector_db.approximate.ivf.index import IVFIndex

    original_search = IVFIndex.search_with_stats

    def wrapped_search(
        self: IVFIndex,
        query: np.ndarray,
        k: int,
    ):
        recorded_k_values.append(k)
        return original_search(self, query, k)

    monkeypatch.setattr(
        IVFIndex,
        "search_with_stats",
        wrapped_search,
    )

    result = benchmark(
        vectors=vectors,
        queries=queries,
        n_clusters=2,
        n_probe=1,
        k_values=(1, 5, 10),
        kmeans_iterations=5,
        seed=42,
    )

    # Three queries × three requested k values.
    assert recorded_k_values == [
        1, 5, 10,
        1, 5, 10,
        1, 5, 10,
    ]

    assert set(result.latency) == {1, 5, 10}

    for k in (1, 5, 10):
        stats = result.latency[k]

        assert stats.p50 >= 0.0
        assert stats.p95 >= stats.p50
        assert stats.p99 >= stats.p95
        assert stats.minimum >= 0.0
        assert stats.maximum >= stats.minimum
        assert stats.mean >= 0.0
        
def test_benchmark_separate_k_searches_preserve_recall() -> None:
    """Separate per-k searches should preserve the recall results."""
    result = benchmark(
        vectors=make_vectors(),
        queries=make_queries(),
        n_clusters=2,
        n_probe=2,
        k_values=(1, 5, 10),
        kmeans_iterations=5,
        seed=42,
    )

    assert result.recall[1] == pytest.approx(1.0)
    assert result.recall[5] == pytest.approx(1.0)
    assert result.recall[10] == pytest.approx(1.0)