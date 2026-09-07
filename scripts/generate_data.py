"""Generate a deterministic clustered vector dataset from YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_OUTPUT_PATH = Path("data/processed/dataset.npz")

Array = npt.NDArray[np.float64]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the project YAML configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def _validate_dataset_config(config: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Extract and validate dataset-generation parameters."""
    dataset = config.get("dataset")
    ivf = config.get("ivf")

    if not isinstance(dataset, dict):
        raise ValueError("Configuration must contain a 'dataset' mapping.")

    if not isinstance(ivf, dict):
        raise ValueError("Configuration must contain an 'ivf' mapping.")

    try:
        n_vectors = int(dataset["n_vectors"])
        n_queries = int(dataset["n_queries"])
        dim = int(dataset["dim"])
        seed = int(dataset["seed"])
        n_clusters = int(ivf["n_clusters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid dataset or IVF configuration.") from exc

    if n_vectors <= 0:
        raise ValueError("dataset.n_vectors must be positive.")

    if n_queries <= 0:
        raise ValueError("dataset.n_queries must be positive.")

    if dim <= 0:
        raise ValueError("dataset.dim must be positive.")

    if n_clusters <= 0:
        raise ValueError("ivf.n_clusters must be positive.")

    if n_clusters > n_vectors:
        raise ValueError("ivf.n_clusters cannot exceed dataset.n_vectors.")

    return n_vectors, n_queries, dim, seed, n_clusters


def _normalize_rows(vectors: Array) -> Array:
    """Normalize rows to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if np.any(norms == 0.0):
        raise ValueError("Generated vectors must not contain zero vectors.")

    return vectors / norms


def generate_dataset(config: dict[str, Any]) -> tuple[Array, Array, npt.NDArray[np.int64]]:
    """Generate clustered database vectors and query vectors.

    Returns
    -------
    tuple
        ``(vectors, queries, ids)`` where:

        - vectors has shape ``(n_vectors, dim)``
        - queries has shape ``(n_queries, dim)``
        - ids contains stable integer IDs ``0..n_vectors-1``

    The same configuration produces the same arrays because all random
    generation uses one explicitly seeded NumPy generator.
    """
    n_vectors, n_queries, dim, seed, n_clusters = _validate_dataset_config(config)

    rng = np.random.default_rng(seed)

    # Cluster centers provide the structure that IVF-Flat will later exploit.
    centers = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_clusters, dim),
    ).astype(np.float64)

    centers = _normalize_rows(centers)

    vector_clusters = rng.integers(
        low=0,
        high=n_clusters,
        size=n_vectors,
    )

    query_clusters = rng.integers(
        low=0,
        high=n_clusters,
        size=n_queries,
    )

    # Keep the noise small enough that vectors remain concentrated around
    # their assigned center while retaining enough variation for search.
    vector_noise = rng.normal(
        loc=0.0,
        scale=0.08,
        size=(n_vectors, dim),
    )

    query_noise = rng.normal(
        loc=0.0,
        scale=0.08,
        size=(n_queries, dim),
    )

    vectors = centers[vector_clusters] + vector_noise
    queries = centers[query_clusters] + query_noise

    vectors = _normalize_rows(vectors)
    queries = _normalize_rows(queries)

    ids = np.arange(n_vectors, dtype=np.int64)

    return vectors, queries, ids


def save_dataset(
    output_path: Path,
    vectors: Array,
    queries: Array,
    ids: npt.NDArray[np.int64],
) -> None:
    """Save generated vectors, queries, and IDs as a compressed NumPy archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        vectors=vectors,
        queries=queries,
        ids=ids,
    )


def generate_from_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Generate and save a dataset using a YAML configuration file."""
    config = load_config(config_path)

    vectors, queries, ids = generate_dataset(config)

    save_dataset(
        output_path=output_path,
        vectors=vectors,
        queries=queries,
        ids=ids,
    )

    return output_path


def main() -> None:
    """Run dataset generation from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate a deterministic clustered vector dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the generated .npz dataset.",
    )

    args = parser.parse_args()

    output_path = generate_from_config(
        config_path=args.config,
        output_path=args.output,
    )

    print(f"Dataset written to {output_path}")


if __name__ == "__main__":
    main()