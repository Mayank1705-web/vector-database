"""Measure memory usage of the dataset, brute-force index, and IVF-Flat index."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_db.approximate.ivf.index import IVFIndex
from vector_db.exact.brute_force import BruteForceIndex


PROCESS = psutil.Process()


def process_memory_bytes() -> int:
    """Return current process resident memory in bytes."""
    return int(PROCESS.memory_info().rss)


def bytes_to_mb(value: int | float) -> float:
    """Convert bytes to mebibytes."""
    return float(value) / (1024.0 * 1024.0)


def measure_dataset_memory(
    vectors: np.ndarray,
) -> dict[str, float | int | str]:
    """Measure the raw in-memory dataset footprint."""
    return {
        "bytes": int(vectors.nbytes),
        "megabytes": bytes_to_mb(vectors.nbytes),
        "vectors": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
    }


def measure_brute_force(
    vectors: np.ndarray,
) -> dict[str, float | int]:
    """Build a brute-force index and measure process memory increase."""
    gc.collect()

    before = process_memory_bytes()

    index = BruteForceIndex()

    for vector_id, vector in enumerate(vectors):
        index.insert(vector_id, vector)

    gc.collect()

    after = process_memory_bytes()
    memory_increase = max(0, after - before)

    return {
        "memory_increase_bytes": int(memory_increase),
        "memory_increase_megabytes": bytes_to_mb(memory_increase),
        "process_memory_before_bytes": int(before),
        "process_memory_after_bytes": int(after),
        "vector_count": int(vectors.shape[0]),
    }


def measure_ivf(
    vectors: np.ndarray,
    n_clusters: int,
    n_probe: int,
    kmeans_iterations: int,
    seed: int,
) -> dict[str, float | int]:
    """Build an IVF-Flat index and measure process memory increase."""
    gc.collect()

    before = process_memory_bytes()

    index = IVFIndex(
        n_clusters=n_clusters,
        n_probe=n_probe,
        kmeans_iterations=kmeans_iterations,
        seed=seed,
    )

    build_start = time.perf_counter()

    index.fit(vectors)

    build_time = time.perf_counter() - build_start

    gc.collect()

    after = process_memory_bytes()
    memory_increase = max(0, after - before)

    return {
        "memory_increase_bytes": int(memory_increase),
        "memory_increase_megabytes": bytes_to_mb(memory_increase),
        "process_memory_before_bytes": int(before),
        "process_memory_after_bytes": int(after),
        "vector_count": int(vectors.shape[0]),
        "n_clusters": int(n_clusters),
        "n_probe": int(n_probe),
        "kmeans_iterations": int(kmeans_iterations),
        "build_time_seconds": float(build_time),
    }


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    dataset_path = ROOT / "data" / "processed" / "dataset.npz"
    output_path = ROOT / "experiments" / "results" / "memory_baseline.json"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with np.load(dataset_path) as dataset:
        vectors = np.asarray(dataset["vectors"], dtype=np.float64)

    if vectors.ndim != 2:
        raise ValueError("Dataset vectors must be a 2-D matrix.")

    n_clusters = int(config["ivf"]["n_clusters"])
    n_probe = int(config["ivf"]["n_probe"])
    kmeans_iterations = int(config["ivf"]["kmeans_iterations"])
    seed = int(config["dataset"]["seed"])

    print("Memory Measurement")
    print("==================")
    print()
    print("Platform")
    print("--------")
    print(f"OS:              {sys.platform}")
    print(f"Process RSS:      {bytes_to_mb(process_memory_bytes()):.2f} MB")
    print()

    print("Dataset")
    print("-------")
    print(f"Vectors:         {vectors.shape[0]}")
    print(f"Dimension:       {vectors.shape[1]}")
    print(f"Dtype:           {vectors.dtype}")
    print(f"Raw data:        {bytes_to_mb(vectors.nbytes):.2f} MB")
    print()

    dataset_result = measure_dataset_memory(vectors)

    print("Brute Force")
    print("-----------")

    brute_force_result = measure_brute_force(vectors)

    print(
        f"Memory increase: "
        f"{brute_force_result['memory_increase_megabytes']:.2f} MB"
    )
    print()

    print("IVF-Flat")
    print("--------")

    ivf_result = measure_ivf(
        vectors=vectors,
        n_clusters=n_clusters,
        n_probe=n_probe,
        kmeans_iterations=kmeans_iterations,
        seed=seed,
    )

    print(f"Clusters:         {n_clusters}")
    print(f"Probes:           {n_probe}")
    print(
        f"Memory increase:  "
        f"{ivf_result['memory_increase_megabytes']:.2f} MB"
    )
    print(
        f"Build time:       "
        f"{ivf_result['build_time_seconds']:.4f} s"
    )

    output = {
        "experiment": "memory_baseline",
        "measurement": {
            "method": "psutil.Process.memory_info().rss",
            "unit": "bytes",
            "platform": sys.platform,
        },
        "dataset": dataset_result,
        "brute_force": brute_force_result,
        "ivf_flat": ivf_result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()