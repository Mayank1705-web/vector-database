"""Tests for IVF inverted lists."""

from __future__ import annotations

import pytest

from vector_db.approximate.ivf.inverted_lists import InvertedLists


def test_creates_empty_clusters() -> None:
    lists = InvertedLists(3)

    assert lists.n_clusters == 3
    assert lists.get(0) == []
    assert lists.get(1) == []
    assert lists.get(2) == []
    assert lists.total_count() == 0


def test_add_and_get_vector_ids() -> None:
    lists = InvertedLists(3)

    lists.add(1, 20)
    lists.add(1, 10)
    lists.add(1, 30)

    assert lists.get(1) == [10, 20, 30]


def test_clusters_are_independent() -> None:
    lists = InvertedLists(3)

    lists.add(0, 10)
    lists.add(1, 20)
    lists.add(2, 30)

    assert lists.get(0) == [10]
    assert lists.get(1) == [20]
    assert lists.get(2) == [30]


def test_duplicate_vector_id_is_stored_once() -> None:
    lists = InvertedLists(2)

    lists.add(0, 42)
    lists.add(0, 42)

    assert lists.get(0) == [42]
    assert lists.count(0) == 1
    assert lists.total_count() == 1


def test_remove_vector_id() -> None:
    lists = InvertedLists(2)

    lists.add(0, 10)
    lists.add(0, 20)

    lists.remove(0, 10)

    assert lists.get(0) == [20]
    assert lists.total_count() == 1


def test_remove_unknown_vector_id_raises() -> None:
    lists = InvertedLists(2)

    with pytest.raises(KeyError, match="not present"):
        lists.remove(0, 99)


def test_clear_cluster() -> None:
    lists = InvertedLists(2)

    lists.add(0, 10)
    lists.add(0, 20)
    lists.add(1, 30)

    lists.clear(0)

    assert lists.get(0) == []
    assert lists.get(1) == [30]
    assert lists.total_count() == 1


def test_count_returns_cluster_size() -> None:
    lists = InvertedLists(2)

    lists.add(0, 10)
    lists.add(0, 20)
    lists.add(1, 30)

    assert lists.count(0) == 2
    assert lists.count(1) == 1


@pytest.mark.parametrize("n_clusters", [0, -1])
def test_invalid_number_of_clusters_is_rejected(
    n_clusters: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        InvertedLists(n_clusters)


def test_non_integer_number_of_clusters_is_rejected() -> None:
    with pytest.raises(TypeError, match="integer"):
        InvertedLists(2.5)


def test_boolean_number_of_clusters_is_rejected() -> None:
    with pytest.raises(TypeError, match="integer"):
        InvertedLists(True)


@pytest.mark.parametrize("cluster_id", [-1, 3])
def test_invalid_cluster_id_is_rejected(cluster_id: int) -> None:
    lists = InvertedLists(3)

    with pytest.raises(IndexError, match="outside"):
        lists.get(cluster_id)


def test_non_integer_cluster_id_is_rejected() -> None:
    lists = InvertedLists(3)

    with pytest.raises(TypeError, match="integer"):
        lists.add(1.5, 10)


def test_boolean_cluster_id_is_rejected() -> None:
    lists = InvertedLists(3)

    with pytest.raises(TypeError, match="integer"):
        lists.add(True, 10)


def test_non_integer_vector_id_is_rejected() -> None:
    lists = InvertedLists(3)

    with pytest.raises(TypeError, match="integer"):
        lists.add(0, 1.5)


def test_boolean_vector_id_is_rejected() -> None:
    lists = InvertedLists(3)

    with pytest.raises(TypeError, match="integer"):
        lists.add(0, True)


def test_get_returns_a_new_list() -> None:
    lists = InvertedLists(2)

    lists.add(0, 10)

    result = lists.get(0)
    result.append(20)

    assert lists.get(0) == [10]