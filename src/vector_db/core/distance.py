"""Distance and similarity functions for vector operations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


VectorMatrix = npt.NDArray[np.float64]


def cosine_similarity(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
) -> float:
    """Return the cosine similarity between two vectors.

    A zero vector has similarity 0.0 with every vector.
    """
    a_array = np.asarray(a, dtype=np.float64)
    b_array = np.asarray(b, dtype=np.float64)

    if a_array.ndim != 1 or b_array.ndim != 1:
        raise ValueError("cosine_similarity expects 1-D vectors.")

    if a_array.shape != b_array.shape:
        raise ValueError("Vectors must have the same dimension.")

    if a_array.size == 0:
        raise ValueError("Vectors must not be empty.")

    if not np.all(np.isfinite(a_array)) or not np.all(np.isfinite(b_array)):
        raise ValueError("Vectors must contain only finite values.")

    norm_a = np.linalg.norm(a_array)
    norm_b = np.linalg.norm(b_array)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a_array, b_array) / (norm_a * norm_b))


def batch_cosine_similarity(
    query: npt.ArrayLike,
    vectors: npt.ArrayLike,
) -> VectorMatrix:
    """Return cosine similarities between one query and many vectors.

    Parameters
    ----------
    query:
        A 1-D query vector with shape (D,).

    vectors:
        A 2-D matrix of vectors with shape (N, D).

    Returns
    -------
    np.ndarray
        A 1-D array of similarities with shape (N,).

    A zero query or zero row vector produces similarity 0.0.
    """
    query_array = np.asarray(query, dtype=np.float64)
    vectors_array = np.asarray(vectors, dtype=np.float64)

    if query_array.ndim != 1:
        raise ValueError("query must be a 1-D vector.")

    if vectors_array.ndim != 2:
        raise ValueError("vectors must be a 2-D matrix.")

    if query_array.size == 0:
        raise ValueError("query must not be empty.")

    if vectors_array.shape[1] != query_array.shape[0]:
        raise ValueError("Query and vectors must have the same dimension.")

    if not np.all(np.isfinite(query_array)):
        raise ValueError("query must contain only finite values.")

    if not np.all(np.isfinite(vectors_array)):
        raise ValueError("vectors must contain only finite values.")

    query_norm = np.linalg.norm(query_array)
    vector_norms = np.linalg.norm(vectors_array, axis=1)

    if query_norm == 0.0:
        return np.zeros(vectors_array.shape[0], dtype=np.float64)

    denominator = query_norm * vector_norms

    similarities = np.zeros(vectors_array.shape[0], dtype=np.float64)

    nonzero = denominator != 0.0

    similarities[nonzero] = (
        vectors_array[nonzero] @ query_array
    ) / denominator[nonzero]

    return similarities


__all__ = [
    "batch_cosine_similarity",
    "cosine_similarity",
]