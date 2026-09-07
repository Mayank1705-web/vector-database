"""Run the vector database benchmark from the project configuration."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow `python scripts\benchmark.py` to work from the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_db.evaluation.benchmark import benchmark


CONFIG_PATH = ROOT / "configs" / "default.yaml"
DATASET_PATH = ROOT / "data" / "processed" / "dataset.npz"
RESULT_PATH = ROOT / "experiments" / "results" / "benchmark_baseline.json"

def main() -> None:
    """Run and print the configured benchmark."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset = np.load(DATASET_PATH)

    vectors = dataset["vectors"]
    queries = dataset["queries"]

    dataset_config = config["dataset"]
    ivf_config = config["ivf"]
    evaluation_config = config["evaluation"]

    start = time.perf_counter()

    result = benchmark(
        vectors=vectors,
        queries=queries,
        n_clusters=ivf_config["n_clusters"],
        n_probe=ivf_config["n_probe"],
        k_values=tuple(evaluation_config["k_values"]),
        kmeans_iterations=ivf_config["kmeans_iterations"],
        seed=dataset_config["seed"],
    )

    elapsed = time.perf_counter() - start
    
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "dataset": {
            "vectors": result.n_vectors,
            "queries": result.n_queries,
            "dimension": result.dimension,
        },
        "ivf": {
            "clusters": result.n_clusters,
            "probes": result.n_probe,
            "kmeans_iterations": ivf_config["kmeans_iterations"],
            "seed": dataset_config["seed"],
        },
        "results": {
            "build_time_seconds": result.build_time,
            "recall": {
                str(k): recall
                for k, recall in result.recall.items()
            },
            "latency_ms": {
                str(k): {
                    "p50": stats.p50,
                    "p95": stats.p95,
                    "p99": stats.p99,
                    "minimum": stats.minimum,
                    "maximum": stats.maximum,
                    "mean": stats.mean,
                }
                for k, stats in result.latency.items()
            },
            "mean_candidate_count": result.mean_candidate_count,
            "mean_candidate_fraction": result.mean_candidate_fraction,
            "total_benchmark_time_seconds": elapsed,
        },
    }
    
    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result_data, file, indent=2)
    
    print(f"Results saved to: {RESULT_PATH}")

    print()
    print("Vector Database Benchmark")
    print("=========================")

    print()
    print("Dataset")
    print("-------")
    print(f"Vectors:             {result.n_vectors}")
    print(f"Queries:             {result.n_queries}")
    print(f"Dimension:           {result.dimension}")

    print()
    print("IVF-Flat")
    print("--------")
    print(f"Clusters:            {result.n_clusters}")
    print(f"Probes:              {result.n_probe}")
    print(f"K-Means iterations:  {ivf_config['kmeans_iterations']}")
    print(f"Build time:          {result.build_time:.6f} s")

    print()
    print("Recall")
    print("------")
    for k, recall in result.recall.items():
        print(f"Recall@{k}:            {recall:.4f}")

    print()
    print("Latency")
    print("-------")

    for k, stats in result.latency.items():
        print(f"Recall@{k} latency:")
        print(f"  P50:               {stats.p50:.6f} ms")
        print(f"  P95:               {stats.p95:.6f} ms")
        print(f"  P99:               {stats.p99:.6f} ms")
        print(f"  Min:               {stats.minimum:.6f} ms")
        print(f"  Max:               {stats.maximum:.6f} ms")
        print(f"  Mean:              {stats.mean:.6f} ms")

    print()
    print("Candidate Reduction")
    print("-------------------")
    print(
        f"Mean candidate count:    "
        f"{result.mean_candidate_count:.2f}"
    )
    print(
        f"Mean candidate fraction: "
        f"{result.mean_candidate_fraction:.4f}"
    )

    print()
    print(f"Total benchmark time: {elapsed:.6f} s")


if __name__ == "__main__":
    main()