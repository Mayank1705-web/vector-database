"""Inverted lists for IVF-Flat cluster membership."""

from __future__ import annotations


class InvertedLists:
    """Store vector IDs grouped by IVF cluster.

    The inverted lists contain only membership information.
    Vector data remains owned by the index/storage layer.
    """

    def __init__(self, n_clusters: int) -> None:
        """Create empty inverted lists for ``n_clusters`` clusters."""
        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise TypeError("n_clusters must be an integer.")

        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        self._lists: list[set[int]] = [
            set() for _ in range(n_clusters)
        ]

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters."""
        return len(self._lists)

    def _validate_cluster_id(self, cluster_id: int) -> None:
        """Validate a cluster ID."""
        if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
            raise TypeError("cluster_id must be an integer.")

        if not 0 <= cluster_id < self.n_clusters:
            raise IndexError(
                f"cluster_id {cluster_id} is outside the valid range "
                f"[0, {self.n_clusters - 1}]."
            )

    @staticmethod
    def _validate_vector_id(vector_id: int) -> None:
        """Validate a vector ID."""
        if not isinstance(vector_id, int) or isinstance(vector_id, bool):
            raise TypeError("vector_id must be an integer.")

    def add(self, cluster_id: int, vector_id: int) -> None:
        """Add a vector ID to a cluster."""
        self._validate_cluster_id(cluster_id)
        self._validate_vector_id(vector_id)

        self._lists[cluster_id].add(vector_id)

    def remove(self, cluster_id: int, vector_id: int) -> None:
        """Remove a vector ID from a cluster."""
        self._validate_cluster_id(cluster_id)
        self._validate_vector_id(vector_id)

        if vector_id not in self._lists[cluster_id]:
            raise KeyError(
                f"vector_id {vector_id} is not present in cluster {cluster_id}."
            )

        self._lists[cluster_id].remove(vector_id)

    def get(self, cluster_id: int) -> list[int]:
        """Return vector IDs belonging to a cluster.

        IDs are returned in sorted order so callers receive deterministic
        results regardless of set iteration order.
        """
        self._validate_cluster_id(cluster_id)
        return sorted(self._lists[cluster_id])

    def count(self, cluster_id: int) -> int:
        """Return the number of vectors in a cluster."""
        self._validate_cluster_id(cluster_id)
        return len(self._lists[cluster_id])

    def total_count(self) -> int:
        """Return the total number of vector memberships."""
        return sum(len(cluster) for cluster in self._lists)

    def clear(self, cluster_id: int) -> None:
        """Remove all vector IDs from a cluster."""
        self._validate_cluster_id(cluster_id)
        self._lists[cluster_id].clear()


__all__ = ["InvertedLists"]