"""Build an IVF-Flat index from the generated dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import yaml

from vector_db.approximate.ivf.index import IVFIndex


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_DATASET_PATH = Path("data/processed/dataset.npz")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the project YAML configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vectors and IDs from the generated dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run scripts/generate_data.py first."
        )

    with np.load(dataset_path) as data:
        if "vectors" not in data:
            raise ValueError("Dataset is missing 'vectors'.")
        if "ids" not in data:
            raise ValueError("Dataset is missing 'ids'.")

        vectors = np.asarray(data["vectors"], dtype=np.float64)
        ids = np.asarray(data["ids"], dtype=np.int64)

    if vectors.ndim != 2:
        raise ValueError("Dataset vectors must be a 2-D array.")

    if ids.ndim != 1:
        raise ValueError("Dataset IDs must be a 1-D array.")

    if vectors.shape[0] != ids.shape[0]:
        raise ValueError("Number of vectors must match number of IDs.")

    if vectors.shape[0] == 0:
        raise ValueError("Dataset must contain at least one vector.")

    if not np.all(np.isfinite(vectors)):
        raise ValueError("Dataset vectors must contain only finite values.")

    return vectors, ids


def build_index(
    config_path: Path = DEFAULT_CONFIG_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> IVFIndex:
    """Build and return an IVF-Flat index from the configured dataset."""
    config = load_config(config_path)

    dataset_config = config.get("dataset")
    ivf_config = config.get("ivf")

    if not isinstance(dataset_config, dict):
        raise ValueError("Configuration must contain a 'dataset' mapping.")

    if not isinstance(ivf_config, dict):
        raise ValueError("Configuration must contain an 'ivf' mapping.")

    n_clusters = int(ivf_config["n_clusters"])
    n_probe = int(ivf_config["n_probe"])
    kmeans_iterations = int(ivf_config["kmeans_iterations"])
    seed = int(dataset_config["seed"])

    vectors, ids = load_dataset(dataset_path)

    expected_n_vectors = int(dataset_config["n_vectors"])
    expected_dim = int(dataset_config["dim"])

    if vectors.shape[0] != expected_n_vectors:
        raise ValueError(
            f"Dataset contains {vectors.shape[0]} vectors, "
            f"expected {expected_n_vectors}."
        )

    if vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Dataset dimension is {vectors.shape[1]}, "
            f"expected {expected_dim}."
        )

    expected_ids = np.arange(expected_n_vectors, dtype=np.int64)

    if not np.array_equal(ids, expected_ids):
        raise ValueError("Dataset IDs must be sequential IDs from 0 to n_vectors-1.")

    index = IVFIndex(
        n_clusters=n_clusters,
        n_probe=n_probe,
        kmeans_iterations=kmeans_iterations,
        seed=seed,
    )

    start = time.perf_counter()
    index.fit(vectors)
    build_time = time.perf_counter() - start

    print("IVF-Flat index build complete.")
    print(f"Vectors indexed: {index.count:,}")
    print(f"Dimension: {vectors.shape[1]}")
    print(f"Clusters: {index.n_clusters}")
    print(f"Probes: {index.n_probe}")
    print(f"Build time: {build_time:.3f} seconds")

    if index.count != vectors.shape[0]:
        raise RuntimeError(
            f"Index contains {index.count} vectors, "
            f"expected {vectors.shape[0]}."
        )

    return index


def main() -> None:
    """Build the IVF-Flat index from the command line."""
    parser = argparse.ArgumentParser(
        description="Build an IVF-Flat index from the generated dataset."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the generated .npz dataset.",
    )

    args = parser.parse_args()

    build_index(
        config_path=args.config,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()