"""Exact brute-force vector index.

This is the project's ground truth: every approximate index (IVF-Flat,
HNSW, ...) is measured against the results this index produces.

Search is O(N * D) per query and storage is O(N * D) -- see design.md
section 4. There is no cleverness here on purpose: correctness first,
speed later, and only once a benchmark says speed is actually needed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vector_db.core.distance import batch_cosine_similarity
from vector_db.core.types import SearchResult
from vector_db.core.vector import validate_dimension, validate_vector

_INITIAL_CAPACITY = 1024
_GROWTH_FACTOR = 2


class BruteForceIndex:
    """Exact nearest-neighbour index over cosine similarity.

    Internal state mirrors design.md section 4:

        vectors: matrix [N, D]
        ids:     array  [N]
        active:  boolean array [N]

    Rows are never physically removed on delete. A deleted row is
    marked inactive and its slot is pushed onto a free list so a
    future insert can reuse it, which keeps delete O(1) and avoids
    reallocating the whole matrix on every deletion.
    """

    def __init__(self) -> None:
        self._dim: int | None = None
        self._capacity = 0
        self._size = 0  # rows ever allocated (active + deleted, excludes free reuse)
        self._vectors: npt.NDArray[np.float64] | None = None
        self._ids: npt.NDArray[np.int64] | None = None
        self._active: npt.NDArray[np.bool_] | None = None
        self._id_to_row: dict[int, int] = {}
        self._free_rows: list[int] = []

    def __len__(self) -> int:
        return len(self._id_to_row)

    def __contains__(self, id: int) -> bool:
        return id in self._id_to_row

    @property
    def dim(self) -> int | None:
        """Vector dimension, fixed by the first insert. None if empty."""
        return self._dim

    # ------------------------------------------------------------------
    # capacity management
    # ------------------------------------------------------------------

    def _ensure_capacity(self, dim: int) -> None:
        """Grow the backing arrays (amortized doubling) if needed."""
        if self._vectors is None:
            capacity = _INITIAL_CAPACITY
            self._vectors = np.zeros((capacity, dim), dtype=np.float64)
            self._ids = np.zeros(capacity, dtype=np.int64)
            self._active = np.zeros(capacity, dtype=bool)
            self._capacity = capacity
            return

        if self._size < self._capacity:
            return

        new_capacity = max(self._capacity * _GROWTH_FACTOR, _INITIAL_CAPACITY)
        new_vectors = np.zeros((new_capacity, dim), dtype=np.float64)
        new_ids = np.zeros(new_capacity, dtype=np.int64)
        new_active = np.zeros(new_capacity, dtype=bool)

        new_vectors[: self._capacity] = self._vectors
        new_ids[: self._capacity] = self._ids
        new_active[: self._capacity] = self._active

        self._vectors = new_vectors
        self._ids = new_ids
        self._active = new_active
        self._capacity = new_capacity

    # ------------------------------------------------------------------
    # insert / delete
    # ------------------------------------------------------------------

    def insert(self, id: int, vector: npt.ArrayLike) -> None:
        """Insert `vector` under `id`.

        The first insert fixes the index's dimension; every later
        insert must match it.

        Raises
        ------
        TypeError
            If `id` is not an int.
        ValueError
            If `id` is already active in the index, or `vector`'s
            dimension does not match the index's dimension.
        """
        if not isinstance(id, int):
            raise TypeError("id must be an int.")

        if id in self._id_to_row:
            raise ValueError(f"id {id} is already active in the index.")

        if self._dim is None:
            array = validate_vector(vector)
            self._dim = array.shape[0]
        else:
            array = validate_dimension(vector, self._dim)

        if self._free_rows:
            row = self._free_rows.pop()
        else:
            self._ensure_capacity(self._dim)
            row = self._size
            self._size += 1

        self._vectors[row] = array
        self._ids[row] = id
        self._active[row] = True
        self._id_to_row[id] = row

    def delete(self, id: int) -> None:
        """Mark the vector stored under `id` inactive.

        The id may be reused by a later insert.

        Raises
        ------
        KeyError
            If `id` is not currently active in the index.
        """
        row = self._id_to_row.pop(id, None)
        if row is None:
            raise KeyError(f"id {id} not found in the index.")

        self._active[row] = False
        self._free_rows.append(row)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: npt.ArrayLike, k: int) -> list[SearchResult]:
        """Return the top-`k` most similar active vectors to `query`.

        Results are sorted by descending cosine similarity, with ties
        broken by ascending id for determinism. If fewer than `k`
        vectors are active, all of them are returned.

        Raises
        ------
        ValueError
            If `k` is not a positive integer, or `query`'s dimension
            does not match the index's dimension.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        if self._dim is None or not self._id_to_row:
            # No vectors have ever been inserted, or none are active.
            # Still validate the query's shape if we have a dimension
            # to validate it against.
            if self._dim is not None:
                validate_dimension(query, self._dim)
            return []

        query_array = validate_dimension(query, self._dim)

        active_mask = self._active[: self._size]
        active_vectors = self._vectors[: self._size][active_mask]
        active_ids = self._ids[: self._size][active_mask]

        scores = batch_cosine_similarity(query_array, active_vectors)

        k_eff = min(k, scores.shape[0])

        # Partial selection: find the k_eff largest scores in O(N),
        # then sort only those k_eff candidates.
        if k_eff < scores.shape[0]:
            top_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
        else:
            top_idx = np.arange(scores.shape[0])

        order = sorted(top_idx, key=lambda i: (-scores[i], int(active_ids[i])))

        return [
            SearchResult(id=int(active_ids[i]), score=float(scores[i]))
            for i in order
        ]


__all__ = ["BruteForceIndex"]
