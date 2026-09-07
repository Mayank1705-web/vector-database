"""Tests for deterministic clustered dataset generation."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.generate_data import generate_dataset, save_dataset


@pytest.fixture
def dataset_config() -> dict:
    """Small deterministic configuration for fast unit tests."""
    return {
        "dataset": {
            "n_vectors": 100,
            "n_queries": 20,
            "dim": 16,
            "seed": 42,
            "source": "synthetic",
        },
        "ivf": {
            "n_clusters": 5,
            "n_probe": 2,
            "kmeans_iterations": 10,
        },
    }


def test_generate_dataset_has_expected_shapes(dataset_config: dict) -> None:
    vectors, queries, ids = generate_dataset(dataset_config)

    assert vectors.shape == (100, 16)
    assert queries.shape == (20, 16)
    assert ids.shape == (100,)


def test_generate_dataset_is_deterministic(dataset_config: dict) -> None:
    first_vectors, first_queries, first_ids = generate_dataset(dataset_config)
    second_vectors, second_queries, second_ids = generate_dataset(dataset_config)

    np.testing.assert_array_equal(first_vectors, second_vectors)
    np.testing.assert_array_equal(first_queries, second_queries)
    np.testing.assert_array_equal(first_ids, second_ids)


def test_different_seed_changes_generated_data(dataset_config: dict) -> None:
    first_vectors, first_queries, _ = generate_dataset(dataset_config)

    dataset_config["dataset"]["seed"] = 123

    second_vectors, second_queries, _ = generate_dataset(dataset_config)

    assert not np.array_equal(first_vectors, second_vectors)
    assert not np.array_equal(first_queries, second_queries)


def test_generated_vectors_are_unit_normalized(dataset_config: dict) -> None:
    vectors, queries, _ = generate_dataset(dataset_config)

    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.linalg.norm(queries, axis=1),
        1.0,
        atol=1e-12,
    )


def test_generated_ids_are_sequential(dataset_config: dict) -> None:
    _, _, ids = generate_dataset(dataset_config)

    np.testing.assert_array_equal(
        ids,
        np.arange(100, dtype=np.int64),
    )


def test_clustered_vectors_are_not_identical(dataset_config: dict) -> None:
    vectors, _, _ = generate_dataset(dataset_config)

    unique_vectors = np.unique(vectors, axis=0)

    assert unique_vectors.shape[0] == vectors.shape[0]


def test_dataset_is_saved_and_can_be_loaded(
    dataset_config: dict,
    tmp_path,
) -> None:
    vectors, queries, ids = generate_dataset(dataset_config)

    output_path = tmp_path / "dataset.npz"

    save_dataset(
        output_path=output_path,
        vectors=vectors,
        queries=queries,
        ids=ids,
    )

    assert output_path.exists()

    loaded = np.load(output_path)

    np.testing.assert_array_equal(loaded["vectors"], vectors)
    np.testing.assert_array_equal(loaded["queries"], queries)
    np.testing.assert_array_equal(loaded["ids"], ids)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_vectors", 0),
        ("n_queries", 0),
        ("dim", 0),
    ],
)
def test_invalid_dataset_dimensions_are_rejected(
    dataset_config: dict,
    field: str,
    value: int,
) -> None:
    dataset_config["dataset"][field] = value

    with pytest.raises(ValueError):
        generate_dataset(dataset_config)


def test_too_many_clusters_are_rejected(dataset_config: dict) -> None:
    dataset_config["ivf"]["n_clusters"] = 101

    with pytest.raises(ValueError, match="cannot exceed"):
        generate_dataset(dataset_config)