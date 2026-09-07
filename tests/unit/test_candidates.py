import numpy as np
import pytest

from vector_db.approximate.ivf.index import IVFIndex
from vector_db.evaluation.candidates import candidate_fraction


def make_vectors() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.0, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float64,
    )


def test_search_with_stats_returns_candidate_count() -> None:
    index = IVFIndex(n_clusters=2, n_probe=1)
    index.fit(make_vectors())

    results, candidate_count = index.search_with_stats(
        np.array([1.0, 0.0]),
        2,
    )

    assert len(results) == 2
    assert candidate_count > 0
    assert candidate_count <= index.count


def test_search_with_stats_all_probes_searches_all_vectors() -> None:
    vectors = make_vectors()

    index = IVFIndex(n_clusters=2, n_probe=2)
    index.fit(vectors)

    _, candidate_count = index.search_with_stats(
        np.array([1.0, 0.0]),
        2,
    )

    assert candidate_count == index.count


def test_search_matches_search_with_stats() -> None:
    index = IVFIndex(n_clusters=2, n_probe=1)
    index.fit(make_vectors())

    query = np.array([1.0, 0.0])

    normal_results = index.search(query, 3)
    stats_results, _ = index.search_with_stats(query, 3)

    assert normal_results == stats_results


def test_candidate_fraction() -> None:
    assert candidate_fraction(500, 50000) == pytest.approx(0.01)


def test_candidate_fraction_all_vectors() -> None:
    assert candidate_fraction(50000, 50000) == pytest.approx(1.0)


def test_candidate_fraction_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="negative"):
        candidate_fraction(-1, 50000)


def test_candidate_fraction_rejects_invalid_total() -> None:
    with pytest.raises(ValueError, match="positive"):
        candidate_fraction(10, 0)