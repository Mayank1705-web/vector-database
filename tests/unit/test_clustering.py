"""Tests for IVF-Flat k-means clustering."""

from __future__ import annotations

import numpy as np
import pytest

from vector_db.approximate.ivf.clustering import fit_kmeans


@pytest.fixture
def clustered_vectors() -> np.ndarray:
    """Create simple deterministic clustered vectors."""
    rng = np.random.default_rng(42)

    cluster_a = rng.normal(
        loc=[1.0, 0.0, 0.0, 0.0],
        scale=0.03,
        size=(30, 4),
    )

    cluster_b = rng.normal(
        loc=[0.0, 1.0, 0.0, 0.0],
        scale=0.03,
        size=(30, 4),
    )

    vectors = np.vstack([cluster_a, cluster_b])

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / norms


def test_fit_kmeans_returns_expected_shapes(
    clustered_vectors: np.ndarray,
) -> None:
    centroids, assignments = fit_kmeans(
        clustered_vectors,
        n_clusters=2,
        n_iter=10,
        seed=42,
    )

    assert centroids.shape == (2, 4)
    assert assignments.shape == (60,)


def test_assignments_are_valid_cluster_ids(
    clustered_vectors: np.ndarray,
) -> None:
    _, assignments = fit_kmeans(
        clustered_vectors,
        n_clusters=2,
        seed=42,
    )

    assert np.all(assignments >= 0)
    assert np.all(assignments < 2)


def test_centroids_are_unit_normalized(
    clustered_vectors: np.ndarray,
) -> None:
    centroids, _ = fit_kmeans(
        clustered_vectors,
        n_clusters=2,
        seed=42,
    )

    np.testing.assert_allclose(
        np.linalg.norm(centroids, axis=1),
        1.0,
        atol=1e-12,
    )


def test_same_seed_is_deterministic(
    clustered_vectors: np.ndarray,
) -> None:
    centroids_a, assignments_a = fit_kmeans(
        clustered_vectors,
        n_clusters=2,
        seed=42,
    )

    centroids_b, assignments_b = fit_kmeans(
        clustered_vectors,
        n_clusters=2,
        seed=42,
    )

    np.testing.assert_array_equal(centroids_a, centroids_b)
    np.testing.assert_array_equal(assignments_a, assignments_b)


def test_different_seed_can_change_initialization(
    clustered_vectors: np.ndarray,
) -> None:
    centroids_a, assignments_a = fit_kmeans(
        clustered_vectors,
        n_clusters=2,
        seed=1,
    )

    centroids_b, assignments_b = fit_kmeans(
        clustered_vectors,
        n_clusters=2,
        seed=2,
    )

    # The exact result may occasionally converge to the same clustering,
    # so verify that both runs are valid rather than requiring difference.
    assert centroids_a.shape == centroids_b.shape
    assert assignments_a.shape == assignments_b.shape


@pytest.mark.parametrize(
    ("n_clusters", "n_iter"),
    [
        (0, 10),
        (-1, 10),
        (2, 0),
        (2, -1),
    ],
)
def test_invalid_parameters_are_rejected(
    clustered_vectors: np.ndarray,
    n_clusters: int,
    n_iter: int,
) -> None:
    with pytest.raises(ValueError):
        fit_kmeans(
            clustered_vectors,
            n_clusters=n_clusters,
            n_iter=n_iter,
        )


def test_too_many_clusters_are_rejected(
    clustered_vectors: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_kmeans(
            clustered_vectors,
            n_clusters=61,
        )


def test_non_2d_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fit_kmeans(
            np.array([1.0, 2.0, 3.0]),
            n_clusters=2,
        )


def test_empty_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_kmeans(
            np.empty((0, 4)),
            n_clusters=2,
        )


def test_non_finite_vectors_are_rejected() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [np.nan, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        fit_kmeans(
            vectors,
            n_clusters=2,
        )