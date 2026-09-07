"""Focused tests for the common VectorIndex API."""

from __future__ import annotations

import pytest

from vector_db.api.index import VectorIndex
from vector_db.core.types import SearchResult


class RecordingBackend:
    """Small fake backend used to verify API delegation."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.results = [
            SearchResult(id=10, score=0.95),
            SearchResult(id=20, score=0.85),
        ]

    def insert(self, id: int, vector) -> None:
        self.calls.append(("insert", id, vector))

    def search(self, query, k: int) -> list[SearchResult]:
        self.calls.append(("search", query, k))
        return self.results

    def delete(self, id: int) -> None:
        self.calls.append(("delete", id))


class MissingSearchBackend:
    """Backend missing the required search method."""

    def insert(self, id: int, vector) -> None:
        pass

    def delete(self, id: int) -> None:
        pass


def test_insert_delegates_to_backend() -> None:
    backend = RecordingBackend()
    index = VectorIndex(backend)

    vector = [1.0, 2.0, 3.0]

    index.insert(7, vector)

    assert len(backend.calls) == 1
    operation, vector_id, received_vector = backend.calls[0]

    assert operation == "insert"
    assert vector_id == 7
    assert received_vector == vector


def test_search_delegates_to_backend_and_returns_results() -> None:
    backend = RecordingBackend()
    index = VectorIndex(backend)

    query = [1.0, 0.0, 0.0]

    results = index.search(query, k=2)

    assert results is backend.results
    assert backend.calls == [
        ("search", query, 2),
    ]


def test_delete_delegates_to_backend() -> None:
    backend = RecordingBackend()
    index = VectorIndex(backend)

    index.delete(7)

    assert backend.calls == [
        ("delete", 7),
    ]


def test_insert_search_delete_all_delegate_to_same_backend() -> None:
    backend = RecordingBackend()
    index = VectorIndex(backend)

    index.insert(1, [1.0, 0.0])
    results = index.search([1.0, 0.0], k=1)
    index.delete(1)

    assert results is backend.results
    assert [call[0] for call in backend.calls] == [
        "insert",
        "search",
        "delete",
    ]


def test_none_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend must not be None"):
        VectorIndex(None)


def test_backend_missing_required_method_is_rejected() -> None:
    with pytest.raises(TypeError, match="backend must implement"):
        VectorIndex(MissingSearchBackend())


def test_exact_index_still_works_through_common_api() -> None:
    from vector_db.exact.brute_force import BruteForceIndex

    index = VectorIndex(BruteForceIndex())

    index.insert(1, [1.0, 0.0, 0.0])
    index.insert(2, [0.0, 1.0, 0.0])
    index.insert(3, [0.9, 0.1, 0.0])

    results = index.search([1.0, 0.0, 0.0], k=2)

    assert [result.id for result in results] == [1, 3]

    index.delete(1)

    results = index.search([1.0, 0.0, 0.0], k=2)

    assert 1 not in [result.id for result in results]