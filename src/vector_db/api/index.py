"""Common API for vector indexes."""

from __future__ import annotations

from typing import Protocol

import numpy.typing as npt

from vector_db.core.types import SearchResult


class IndexBackend(Protocol):
    """Protocol implemented by concrete vector-index backends."""

    def insert(self, id: int, vector: npt.ArrayLike) -> None:
        """Insert a vector under an ID."""

    def search(
        self,
        query: npt.ArrayLike,
        k: int,
    ) -> list[SearchResult]:
        """Return the top-k nearest vectors."""

    def delete(self, id: int) -> None:
        """Delete the vector associated with an ID."""


class VectorIndex:
    """Common user-facing API for all vector-index implementations.

    The wrapper intentionally contains no indexing or search algorithm.
    Concrete backends such as BruteForceIndex and IVF-Flat own those
    implementation details.
    """

    def __init__(self, backend: IndexBackend) -> None:
        """Create a common API around a concrete index backend."""
        if backend is None:
            raise ValueError("backend must not be None.")

        required_methods = ("insert", "search", "delete")

        for method_name in required_methods:
            if not callable(getattr(backend, method_name, None)):
                raise TypeError(
                    f"backend must implement callable '{method_name}'."
                )

        self._backend = backend

    def insert(self, id: int, vector: npt.ArrayLike) -> None:
        """Insert a vector into the underlying index."""
        self._backend.insert(id, vector)

    def search(
        self,
        query: npt.ArrayLike,
        k: int,
    ) -> list[SearchResult]:
        """Search the underlying index for the top-k nearest vectors."""
        return self._backend.search(query, k)

    def delete(self, id: int) -> None:
        """Delete a vector from the underlying index."""
        self._backend.delete(id)


__all__ = [
    "IndexBackend",
    "VectorIndex",
]