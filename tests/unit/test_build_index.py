from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.build_index import build_index, load_dataset


def create_config(path: Path, n_vectors: int, dim: int) -> None:
    """Create a minimal configuration for testing."""
    config = {
        "dataset": {
            "n_vectors": n_vectors,
            "n_queries": 2,
            "dim": dim,
            "seed": 42,
        },
        "ivf": {
            "n_clusters": 2,
            "n_probe": 1,
            "kmeans_iterations": 3,
        },
    }

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)


def create_dataset(path: Path, n_vectors: int, dim: int) -> None:
    """Create a small deterministic dataset for testing."""
    rng = np.random.default_rng(42)

    vectors = rng.normal(
        size=(n_vectors, dim),
    ).astype(np.float64)

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    ids = np.arange(n_vectors, dtype=np.int64)

    queries = vectors[:2].copy()

    np.savez_compressed(
        path,
        vectors=vectors,
        queries=queries,
        ids=ids,
    )


def test_load_dataset_returns_vectors_and_ids(tmp_path: Path) -> None:
    """Dataset loader returns the stored vectors and IDs."""
    dataset_path = tmp_path / "dataset.npz"
    create_dataset(dataset_path, n_vectors=6, dim=4)

    vectors, ids = load_dataset(dataset_path)

    assert vectors.shape == (6, 4)
    assert ids.shape == (6,)
    assert np.array_equal(ids, np.arange(6))


def test_build_index_creates_trained_index(tmp_path: Path) -> None:
    """Build pipeline creates a trained IVF index."""
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "dataset.npz"

    create_config(config_path, n_vectors=6, dim=4)
    create_dataset(dataset_path, n_vectors=6, dim=4)

    index = build_index(
        config_path=config_path,
        dataset_path=dataset_path,
    )

    assert index.is_trained
    assert index.count == 6


def test_build_index_uses_configured_parameters(tmp_path: Path) -> None:
    """Build pipeline applies IVF configuration values."""
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "dataset.npz"

    create_config(config_path, n_vectors=8, dim=4)
    create_dataset(dataset_path, n_vectors=8, dim=4)

    index = build_index(
        config_path=config_path,
        dataset_path=dataset_path,
    )

    assert index.n_clusters == 2
    assert index.n_probe == 1


def test_build_index_rejects_wrong_vector_count(tmp_path: Path) -> None:
    """Build pipeline rejects a dataset with the wrong vector count."""
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "dataset.npz"

    create_config(config_path, n_vectors=8, dim=4)
    create_dataset(dataset_path, n_vectors=6, dim=4)

    with pytest.raises(ValueError, match="vectors"):
        build_index(
            config_path=config_path,
            dataset_path=dataset_path,
        )


def test_build_index_rejects_wrong_dimension(tmp_path: Path) -> None:
    """Build pipeline rejects a dataset with the wrong dimension."""
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "dataset.npz"

    create_config(config_path, n_vectors=6, dim=8)
    create_dataset(dataset_path, n_vectors=6, dim=4)

    with pytest.raises(ValueError, match="dimension"):
        build_index(
            config_path=config_path,
            dataset_path=dataset_path,
        )


def test_build_index_rejects_invalid_ids(tmp_path: Path) -> None:
    """Build pipeline rejects IDs that are not 0..n_vectors-1."""
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "dataset.npz"

    create_config(config_path, n_vectors=6, dim=4)
    create_dataset(dataset_path, n_vectors=6, dim=4)

    np.savez_compressed(
        dataset_path,
        vectors=np.load(dataset_path)["vectors"],
        queries=np.zeros((2, 4)),
        ids=np.arange(1, 7, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="IDs"):
        build_index(
            config_path=config_path,
            dataset_path=dataset_path,
        )


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    """Dataset loader rejects a missing dataset."""
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.npz")