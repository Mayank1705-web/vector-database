import numpy as np
import pytest

from vector_db.approximate.ivf.index import IVFIndex
from vector_db.exact.brute_force import BruteForceIndex


def make_vectors():
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )


def make_trained_index(n_probe=2):
    index = IVFIndex(n_clusters=2, n_probe=n_probe)

    training_vectors = make_vectors()
    index.fit(training_vectors)

    return index


def test_empty_index_returns_empty_results():
    index = IVFIndex(n_clusters=2, n_probe=1)

    assert index.search([1.0, 0.0], 3) == []


def test_fit_creates_trained_index():
    index = IVFIndex(n_clusters=2, n_probe=1)

    assert not index.is_trained

    index.fit(make_vectors())

    assert index.is_trained
    assert index.count == 4


def test_insert_and_search():
    index = make_trained_index()

    index.insert(10, [0.7, 0.7])

    results = index.search([0.7, 0.7], 1)

    assert len(results) == 1
    assert results[0].id == 10
    assert results[0].score == pytest.approx(1.0)


def test_duplicate_id_is_rejected():
    index = make_trained_index()

    index.insert(10, [1.0, 0.0])

    with pytest.raises(ValueError):
        index.insert(10, [0.0, 1.0])


def test_dimension_mismatch_is_rejected():
    index = make_trained_index()

    with pytest.raises(ValueError):
        index.insert(10, [1.0, 0.0, 0.0])


def test_delete_removes_vector_from_search():
    index = make_trained_index()

    index.insert(10, [1.0, 0.0])

    index.delete(10)

    results = index.search([1.0, 0.0], 10)

    assert 10 not in [result.id for result in results]


def test_delete_unknown_id_raises():
    index = make_trained_index()

    with pytest.raises(KeyError):
        index.delete(999)


def test_k_must_be_positive():
    index = make_trained_index()

    with pytest.raises(ValueError):
        index.search([1.0, 0.0], 0)


def test_n_probe_must_be_valid():
    with pytest.raises(ValueError):
        IVFIndex(n_clusters=4, n_probe=5)


def test_n_probe_must_be_positive():
    with pytest.raises(ValueError):
        IVFIndex(n_clusters=4, n_probe=0)


def test_invalid_n_clusters():
    with pytest.raises(ValueError):
        IVFIndex(n_clusters=0, n_probe=1)


def test_insert_before_training_is_rejected():
    index = IVFIndex(n_clusters=2, n_probe=1)

    with pytest.raises(RuntimeError):
        index.insert(1, [1.0, 0.0])


def test_search_dimension_mismatch_is_rejected():
    index = make_trained_index()

    with pytest.raises(ValueError):
        index.search([1.0, 0.0, 0.0], 1)


def test_ivf_matches_brute_force_when_all_clusters_are_probed():
    vectors = make_vectors()

    ivf = IVFIndex(n_clusters=2, n_probe=2)
    ivf.fit(vectors)

    brute_force = BruteForceIndex()

    for vector_id, vector in enumerate(vectors):
        brute_force.insert(vector_id, vector)

    query = np.array([0.8, 0.2])

    ivf_results = ivf.search(query, 4)
    brute_results = brute_force.search(query, 4)

    assert [result.id for result in ivf_results] == [
        result.id for result in brute_results
    ]

    np.testing.assert_allclose(
        [result.score for result in ivf_results],
        [result.score for result in brute_results],
    )