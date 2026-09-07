"""K-means clustering for IVF-Flat."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vector_db.core.distance import batch_cosine_similarity
from vector_db.core.vector import validate_vector

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _validate_inputs(
    vectors: npt.ArrayLike,
    n_clusters: int,
    n_iter: int,
) -> Array:
    """Validate k-means inputs."""
    data = np.asarray(vectors, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("vectors must be a 2-D array.")

    if data.shape[0] == 0:
        raise ValueError("vectors must not be empty.")

    if data.shape[1] == 0:
        raise ValueError("vectors must have at least one dimension.")

    if not np.all(np.isfinite(data)):
        raise ValueError("vectors must contain only finite values.")

    if not isinstance(n_clusters, (int, np.integer)):
        raise TypeError("n_clusters must be an integer.")

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")

    if n_clusters > data.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of vectors.")

    if not isinstance(n_iter, (int, np.integer)):
        raise TypeError("n_iter must be an integer.")

    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    return np.array(data, dtype=np.float64, copy=True)


def _normalize_rows(vectors: Array) -> Array:
    """Normalize vectors row-wise."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if np.any(norms == 0.0):
        raise ValueError("vectors must not contain zero vectors.")

    return vectors / norms


def fit_kmeans(
    vectors: npt.ArrayLike,
    n_clusters: int,
    n_iter: int = 10,
    seed: int = 0,
) -> tuple[Array, IntArray]:
    """Fit cosine-similarity k-means and return centroids and assignments.

    Parameters
    ----------
    vectors:
        Matrix with shape ``(n_vectors, dim)``.
    n_clusters:
        Number of clusters.
    n_iter:
        Number of k-means iterations.
    seed:
        Random seed used for centroid initialization.

    Returns
    -------
    tuple
        ``(centroids, assignments)`` where:

        - centroids has shape ``(n_clusters, dim)``
        - assignments has shape ``(n_vectors,)``
    """
    data = _validate_inputs(vectors, n_clusters, n_iter)

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    data = _normalize_rows(data)

    rng = np.random.default_rng(seed)

    initial_indices = rng.choice(
        data.shape[0],
        size=n_clusters,
        replace=False,
    )

    centroids = np.array(
        data[initial_indices],
        dtype=np.float64,
        copy=True,
    )

    assignments = np.full(
        data.shape[0],
        -1,
        dtype=np.int64,
    )

    for _ in range(n_iter):
        similarities = data @ centroids.T

        new_assignments = np.argmax(similarities, axis=1).astype(np.int64)

        new_centroids = np.empty_like(centroids)

        for cluster_id in range(n_clusters):
            members = data[new_assignments == cluster_id]

            if members.shape[0] == 0:
                # Keep the previous centroid when a cluster becomes empty.
                new_centroids[cluster_id] = centroids[cluster_id]
            else:
                centroid = np.mean(members, axis=0)
                norm = np.linalg.norm(centroid)

                if norm == 0.0:
                    new_centroids[cluster_id] = centroids[cluster_id]
                else:
                    new_centroids[cluster_id] = centroid / norm

        centroids = new_centroids

        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break

        assignments = new_assignments

    return centroids, assignments


__all__ = ["fit_kmeans"]