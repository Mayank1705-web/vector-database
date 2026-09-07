"""IVF-Flat approximate vector index."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vector_db.approximate.ivf.clustering import fit_kmeans
from vector_db.approximate.ivf.inverted_lists import InvertedLists
from vector_db.core.distance import batch_cosine_similarity
from vector_db.core.types import SearchResult
from vector_db.core.vector import validate_vector


class IVFIndex:
    """Inverted File index with exact vector scoring.

    IVF is used only to reduce the number of candidate vectors.
    Similarity between the query and candidate vectors is calculated
    exactly using cosine similarity.
    """

    def __init__(
        self,
        n_clusters: int,
        n_probe: int = 1,
        kmeans_iterations: int = 10,
        seed: int = 42,
    ) -> None:
        """Create an untrained IVF-Flat index."""
        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise TypeError("n_clusters must be an integer.")

        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        if not isinstance(n_probe, int) or isinstance(n_probe, bool):
            raise TypeError("n_probe must be an integer.")

        if n_probe <= 0:
            raise ValueError("n_probe must be positive.")

        if n_probe > n_clusters:
            raise ValueError("n_probe must not exceed n_clusters.")

        if (
            not isinstance(kmeans_iterations, int)
            or isinstance(kmeans_iterations, bool)
        ):
            raise TypeError("kmeans_iterations must be an integer.")

        if kmeans_iterations <= 0:
            raise ValueError("kmeans_iterations must be positive.")

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer.")

        self._n_clusters = n_clusters
        self._n_probe = n_probe
        self._kmeans_iterations = kmeans_iterations
        self._seed = seed

        self._centroids: np.ndarray | None = None
        self._inverted_lists = InvertedLists(n_clusters)

        self._vectors: dict[int, np.ndarray] = {}
        self._id_to_cluster: dict[int, int] = {}

        self._dimension: int | None = None

    @property
    def n_clusters(self) -> int:
        """Return the number of IVF clusters."""
        return self._n_clusters

    @property
    def n_probe(self) -> int:
        """Return the number of clusters searched per query."""
        return self._n_probe

    @property
    def is_trained(self) -> bool:
        """Return whether the index has trained centroids."""
        return self._centroids is not None

    @property
    def count(self) -> int:
        """Return the number of stored vectors."""
        return len(self._vectors)

    def fit(
        self,
        vectors: npt.ArrayLike,
    ) -> None:
        """Train IVF centroids and assign existing vectors to clusters."""
        data = np.asarray(vectors, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError("vectors must be a 2-D array.")

        if data.shape[0] < self._n_clusters:
            raise ValueError(
                "Number of vectors must be at least n_clusters."
            )

        if data.shape[1] == 0:
            raise ValueError("vectors must have a non-zero dimension.")

        if not np.all(np.isfinite(data)):
            raise ValueError("vectors must contain only finite values.")

        centroids, _ = fit_kmeans(
            data,
            n_clusters=self._n_clusters,
            n_iter=self._kmeans_iterations,
            seed=self._seed,
        )
        
        self._centroids = centroids

        self._centroids = centroids

        if self._dimension is None:
            self._dimension = data.shape[1]
        elif self._dimension != data.shape[1]:
            raise ValueError("Vector dimension does not match the index.")

        self._inverted_lists = InvertedLists(self._n_clusters)
        self._id_to_cluster.clear()

        # Replace stored vectors with the supplied training vectors.
        self._vectors = {
            vector_id: np.array(vector, dtype=np.float64, copy=True)
            for vector_id, vector in enumerate(data)
        }

        assignments = np.argmax(data @ centroids.T, axis=1)

        for vector_id, cluster_id in enumerate(assignments):
            cluster = int(cluster_id)
            self._id_to_cluster[vector_id] = cluster
            self._inverted_lists.add(cluster, vector_id)

    def insert(self, id: int, vector: npt.ArrayLike) -> None:
        """Insert a vector into the trained IVF index."""
        if not isinstance(id, int) or isinstance(id, bool):
            raise TypeError("id must be an integer.")

        if id in self._vectors:
            raise ValueError(f"Vector ID {id} already exists.")

        if not self.is_trained:
            raise RuntimeError(
                "IVFIndex must be trained with fit() before insert()."
            )

        validated = validate_vector(vector)

        if self._dimension is None:
            self._dimension = validated.shape[0]

        if validated.shape[0] != self._dimension:
            raise ValueError(
                f"Vector dimension {validated.shape[0]} does not match "
                f"expected dimension {self._dimension}."
            )

        similarities = validated @ self._centroids.T
        cluster_id = int(np.argmax(similarities))

        self._vectors[id] = np.array(
            validated,
            dtype=np.float64,
            copy=True,
        )

        self._id_to_cluster[id] = cluster_id
        self._inverted_lists.add(cluster_id, id)

    def delete(self, id: int) -> None:
        """Delete a vector from the IVF index."""
        if not isinstance(id, int) or isinstance(id, bool):
            raise TypeError("id must be an integer.")

        if id not in self._vectors:
            raise KeyError(f"Vector ID {id} does not exist.")

        cluster_id = self._id_to_cluster[id]

        self._inverted_lists.remove(cluster_id, id)

        del self._vectors[id]
        del self._id_to_cluster[id]

    def search(
        self,
        query: npt.ArrayLike,
        k: int,
    ) -> list[SearchResult]:
        """Return the top-k nearest vectors using IVF candidate search."""
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be positive.")

        validated_query = validate_vector(query)

        if self._dimension is not None:
            if validated_query.shape[0] != self._dimension:
                raise ValueError(
                    f"Query dimension {validated_query.shape[0]} does not "
                    f"match expected dimension {self._dimension}."
                )

        if not self._vectors:
            return []

        if not self.is_trained:
            raise RuntimeError(
                "IVFIndex must be trained before searching."
            )

        # Find the closest n_probe centroids.
        centroid_scores = validated_query @ self._centroids.T

        probe_clusters = np.argpartition(
            -centroid_scores,
            self._n_probe - 1,
        )[: self._n_probe]

        # Collect candidate IDs.
        candidate_ids: list[int] = []

        for cluster_id in probe_clusters:
            candidate_ids.extend(
                self._inverted_lists.get(int(cluster_id))
            )

        if not candidate_ids:
            return []

        # Retrieve candidate vectors.
        candidate_vectors = np.vstack(
            [self._vectors[vector_id] for vector_id in candidate_ids]
        )

        # Exact similarity scoring.
        scores = batch_cosine_similarity(
            validated_query,
            candidate_vectors,
        )

        results = [
            SearchResult(
                id=vector_id,
                score=float(score),
            )
            for vector_id, score in zip(candidate_ids, scores)
        ]

        # Deterministic ordering:
        # highest score first, then lowest ID.
        results.sort(
            key=lambda result: (-result.score, result.id)
        )

        return results[:k]

    def search_with_stats(
        self,
        query: npt.ArrayLike,
        k: int,
    ) -> tuple[list[SearchResult], int]:
        """Search IVF and return results together with candidate count.

        The candidate count is the number of vectors from the selected
        probe clusters that receive exact cosine-similarity scoring.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer.")

        if k <= 0:
            raise ValueError("k must be positive.")

        validated_query = validate_vector(query)

        if self._dimension is not None:
            if validated_query.shape[0] != self._dimension:
                raise ValueError(
                    f"Query dimension {validated_query.shape[0]} does not "
                    f"match expected dimension {self._dimension}."
                )

        if not self._vectors:
            return [], 0

        if not self.is_trained:
            raise RuntimeError(
                "IVFIndex must be trained before searching."
            )

        # Find the closest n_probe centroids.
        centroid_scores = validated_query @ self._centroids.T

        probe_clusters = np.argpartition(
            -centroid_scores,
            self._n_probe - 1,
        )[: self._n_probe]

        # Collect candidate IDs.
        candidate_ids: list[int] = []

        for cluster_id in probe_clusters:
            candidate_ids.extend(
                self._inverted_lists.get(int(cluster_id))
            )

        candidate_count = len(candidate_ids)

        if not candidate_ids:
            return [], 0

        # Retrieve candidate vectors.
        candidate_vectors = np.vstack(
            [self._vectors[vector_id] for vector_id in candidate_ids]
        )

        # Exact similarity scoring.
        scores = batch_cosine_similarity(
            validated_query,
            candidate_vectors,
        )

        results = [
            SearchResult(
                id=vector_id,
                score=float(score),
            )
            for vector_id, score in zip(candidate_ids, scores)
        ]

        # Deterministic ordering.
        results.sort(
            key=lambda result: (-result.score, result.id)
        )

        return results[:k], candidate_count

__all__ = ["IVFIndex"]