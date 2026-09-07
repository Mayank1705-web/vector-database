import numpy as np
import pytest

from vector_db.exact.brute_force import BruteForceIndex


def make_index():
    index = BruteForceIndex()
    index.insert(1, [1.0, 0.0])
    index.insert(2, [0.0, 1.0])
    index.insert(3, [0.9, 0.1])
    return index


def test_search_returns_correct_top_k():
    index = make_index()
    results = index.search([1.0, 0.0], k=2)
    ids = [r.id for r in results]
    assert ids == [1, 3]  # id 1 is exact match, id 3 is next closest


def test_search_scores_are_descending():
    index = make_index()
    results = index.search([1.0, 0.0], k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_available_vectors():
    index = make_index()
    results = index.search([1.0, 0.0], k=100)
    assert len(results) == 3


def test_empty_index_returns_empty_list():
    index = BruteForceIndex()
    assert index.search([1.0, 0.0], k=5) == []


def test_delete_removes_vector_from_results():
    index = make_index()
    index.delete(1)
    results = index.search([1.0, 0.0], k=3)
    ids = [r.id for r in results]
    assert 1 not in ids
    assert len(index) == 2


def test_delete_unknown_id_raises():
    index = make_index()
    with pytest.raises(KeyError):
        index.delete(999)


def test_duplicate_active_id_raises():
    index = make_index()
    with pytest.raises(ValueError):
        index.insert(1, [1.0, 1.0])


def test_reinsert_after_delete_is_allowed():
    index = make_index()
    index.delete(1)
    index.insert(1, [0.5, 0.5])
    results = index.search([0.5, 0.5], k=1)
    assert results[0].id == 1


def test_invalid_dimension_raises():
    index = make_index()
    with pytest.raises(ValueError):
        index.search([1.0, 0.0, 0.0], k=1)


def test_zero_vector_handling():
    index = BruteForceIndex()
    index.insert(1, [0.0, 0.0])
    index.insert(2, [1.0, 0.0])
    results = index.search([1.0, 0.0], k=2)
    scores = {r.id: r.score for r in results}
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx(1.0)


def test_negative_and_duplicate_similar_vectors():
    index = BruteForceIndex()
    index.insert(1, [1.0, 1.0])
    index.insert(2, [1.0, 1.0])  # duplicate vector, different id
    index.insert(3, [-1.0, -1.0])  # opposite vector
    results = index.search([1.0, 1.0], k=3)
    ids = [r.id for r in results]
    assert ids[0] in (1, 2) and ids[1] in (1, 2)
    assert ids[2] == 3
    assert results[2].score == pytest.approx(-1.0)


def test_k_must_be_positive():
    index = make_index()
    with pytest.raises(ValueError):
        index.search([1.0, 0.0], k=0)