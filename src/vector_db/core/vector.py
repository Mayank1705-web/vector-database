"""Core vector validation and normalization utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


VectorArray = npt.NDArray[np.float64]


def validate_vector(vector: npt.ArrayLike) -> VectorArray:
    """Convert and validate a vector.

    Parameters
    ----------
    vector:
        A 1-D numeric vector represented as a list, tuple, or NumPy array.

    Returns
    -------
    np.ndarray
        A copied, read-only float64 NumPy array.

    Raises
    ------
    ValueError
        If the vector is not 1-D, is empty, or contains non-finite values.
    """
    array = np.asarray(vector, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("Vector must be 1-D.")

    if array.size == 0:
        raise ValueError("Vector must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError("Vector must contain only finite values.")

    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)

    return result


def validate_dimension(
    vector: npt.ArrayLike,
    expected_dim: int,
) -> VectorArray:
    """Validate a vector and ensure it has the expected dimension."""
    if expected_dim <= 0:
        raise ValueError("expected_dim must be positive.")

    result = validate_vector(vector)

    if result.shape[0] != expected_dim:
        raise ValueError(
            f"Vector dimension {result.shape[0]} does not match "
            f"expected dimension {expected_dim}."
        )

    return result


__all__ = [
    "VectorArray",
    "validate_dimension",
    "validate_vector",
]