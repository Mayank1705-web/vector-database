"""Measure the effect of IVF-Flat n_probe on recall, latency, and candidates."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_db.evaluation.benchmark import benchmark


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    dataset_path = ROOT / "data" / "processed" / "dataset.npz"
    output_path = ROOT / "experiments" / "results" / "n_probe_experiment.json"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with np.load(dataset_path) as dataset:
        vectors = np.asarray(dataset["vectors"], dtype=np.float64)
        queries = np.asarray(dataset["queries"], dtype=np.float64)

    n_clusters = int(config["ivf"]["n_clusters"])
    kmeans_iterations = int(config["ivf"]["kmeans_iterations"])
    seed = int(config["dataset"]["seed"])
    k_values = [int(value) for value in config["evaluation"]["k_values"]]

    probe_values = [1, 2, 4, 8, 16, 32, 64, 100]

    results = []

    total_start = time.perf_counter()

    for n_probe in probe_values:
        print(f"Running n_probe={n_probe}...")

        result = benchmark(
            vectors=vectors,
            queries=queries,
            n_clusters=n_clusters,
            n_probe=n_probe,
            k_values=k_values,
            kmeans_iterations=kmeans_iterations,
            seed=seed,
        )

        row = {
            "n_probe": n_probe,
            "build_time_seconds": result.build_time,
            "recall": {
                str(k): result.recall[k]
                for k in k_values
            },
            "latency_ms": {
                str(k): {
                    "p50": result.latency[k].p50 * 1000.0,
                    "p95": result.latency[k].p95 * 1000.0,
                    "p99": result.latency[k].p99 * 1000.0,
                    "minimum": result.latency[k].minimum * 1000.0,
                    "maximum": result.latency[k].maximum * 1000.0,
                    "mean": result.latency[k].mean * 1000.0,
                }
                for k in k_values
            },
            "mean_candidate_count": result.mean_candidate_count,
            "mean_candidate_fraction": result.mean_candidate_fraction,
        }

        results.append(row)

        print(
            f"  Recall@10={result.recall.get(10, 0.0):.4f}, "
            f"P95={result.latency[max(k_values)].p95 * 1000.0:.4f} ms, "
            f"candidates={result.mean_candidate_count:.2f} "
            f"({result.mean_candidate_fraction:.4%})"
        )

    total_time = time.perf_counter() - total_start

    output = {
        "experiment": "n_probe",
        "dataset": {
            "vectors": vectors.shape[0],
            "queries": queries.shape[0],
            "dimension": vectors.shape[1],
        },
        "ivf": {
            "clusters": n_clusters,
            "kmeans_iterations": kmeans_iterations,
            "seed": seed,
        },
        "probe_values": probe_values,
        "results": results,
        "total_experiment_time_seconds": total_time,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()