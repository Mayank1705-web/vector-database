# Vector Database From Scratch

A learning-focused vector database implemented from first principles in Python and NumPy.

The goal of this project is to understand how vector databases work internally by implementing the core storage, similarity search, approximate indexing, evaluation, and benchmarking components without relying on existing vector-search/ANN libraries.

---

## Features

* Exact brute-force cosine similarity search
* IVF-Flat approximate nearest-neighbor index
* K-means clustering implemented from scratch
* Inverted lists implemented from scratch
* Insert, search, and delete operations
* Tombstone-style deletion through inactive/free rows in the exact index
* Common index API
* Recall@1, Recall@5, and Recall@10 evaluation
* P50, P95, and P99 latency measurements
* Candidate-count and candidate-fraction measurements
* Deterministic clustered synthetic dataset generation
* `n_probe` parameter experiments
* `n_clusters` parameter experiments
* Process RSS memory measurement
* Small semantic-search demonstration
* Automated unit, integration, and benchmark tests

---

## Project Goals

This project is intentionally educational.

Instead of using an existing vector database or ANN implementation, the important pieces are implemented manually so that the relationship between:

* vectors,
* cosine similarity,
* clustering,
* inverted lists,
* candidate selection,
* approximate search,
* recall,
* and latency

can be observed directly.

The exact brute-force implementation is used as the **correctness oracle** when evaluating IVF-Flat.

---

# Architecture

```text
                         Query Vector
                              |
                              v
                    +-------------------+
                    |    VectorIndex    |
                    |    Common API     |
                    +---------+---------+
                              |
               +--------------+--------------+
               |                             |
               v                             v
      +-------------------+         +-------------------+
      |  BruteForceIndex  |         |     IVF-Flat      |
      |   Exact Search    |         | Approximate Search|
      +---------+---------+         +---------+---------+
                |                             |
                |                             v
                |                    +-------------------+
                |                    |  K-means Centroids|
                |                    +---------+---------+
                |                              |
                |                              v
                |                    +-------------------+
                |                    |  Inverted Lists   |
                |                    +---------+---------+
                |                              |
                |                              v
                |                    Candidate Vectors
                |                              |
                +--------------+---------------+
                               |
                               v
                    Cosine Similarity
                               |
                               v
                         Top-k Results
```

---

# Repository Structure

```text
vector-database/
|--- configs/
│   |--- default.yaml
│
|--- docs/
│   |--- Constraints.md
│   |--- Decisions.md
│   |--- Flow.md
│   |--- Handover.md
│   |--- ROLLBACK.md
│   |--- TEST_CHECKLIST.md
│   |--- features/
│       |--- FEATURE-vector-index.md
│
|--- experiments/
│   |--- notebooks/
│   |--- results/
│
|--- scripts/
│   |--- benchmark.py
│   |--- build_index.py
│   |--- experiment_n_clusters.py
│   |--- experiment_n_probe.py
│   |--- generate_data.py
│   |--- measure_memory.py
│   |--- semantic_search_demo.py
│
|--- src/
│   |--- vector_db/
│       |--- api/
│       |--- approximate/
│       │   |--- ivf/
│       |--- core/
│       |--- evaluation/
│       |--- exact/
│       |--- storage/
│
|--- tests/
│   |--- benchmarks/
│   |--- integration/
│   |--- unit/
│
|--- architecture.md
|--- design.md
|--- phases.md
|--- PRD.md
|--- pyproject.toml
|--- README.md
```

---

# Requirements

* Python 3.10+
* NumPy
* PyYAML
* pytest
* psutil for the cross-platform process-memory measurement script

The project is configured as a Python package using `pyproject.toml`.

---

# Installation

## 1. Clone the repository

```powershell
git clone https://github.com/Mayank1705-web/vector-database.git
cd vector-database
```

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install the project

```powershell
python -m pip install -e .
```

Install development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

If `psutil` is not already installed:

```powershell
python -m pip install psutil
```

---

# Configuration

The default configuration is in:

```text
configs/default.yaml
```

Current benchmark configuration:

```yaml
dataset:
  n_vectors: 50000
  n_queries: 500
  dim: 128
  seed: 42
  source: synthetic

ivf:
  n_clusters: 100
  n_probe: 8
  kmeans_iterations: 10

evaluation:
  k_values: [1, 5, 10]
  latency_percentiles: [50, 95, 99]
```

The experiments independently vary `n_probe` and `n_clusters`.

---

# Dataset

The benchmark uses a **deterministic clustered synthetic dataset**.

The dataset contains:

* 50,000 vectors
* 500 query vectors
* 128 dimensions
* deterministic random seed: 42
* normalized vectors
* clustered vector distribution

The data generator creates normalized cluster centers and generates vectors around those centers with controlled Gaussian noise.

Generate the dataset with:

```powershell
python scripts\generate_data.py
```

The generated dataset is written to:

```text
data\processed\dataset.npz
```

Generated benchmark data is not treated as source code and is ignored by Git.

---

# What Is Mocked / Synthetic?

This distinction is important.

## Synthetic dataset

The benchmark vectors are **not real-world production embeddings**.

They are deterministic synthetic vectors generated to have a clustered structure that is useful for evaluating IVF-Flat.

This makes experiments:

* reproducible,
* controlled,
* fast to regenerate,
* and useful for studying approximate search behavior.

However, results on this dataset should **not** be interpreted as representative of every real embedding workload.

---

## Semantic embedding in the demo

The semantic-search demo does **not** use a real transformer or production embedding model.

`semantic_search_demo.py` contains a small deterministic text-to-vector function based on hashed word/character features.

It is intentionally lightweight and exists to demonstrate the pipeline:

```text
Text
 ->
Toy deterministic embedding
 ->
Own vector database
 ->
Similarity search
 ->
Best matching statement
```

It is **not a production-quality embedding model**.

The demo therefore illustrates the vector database/search pipeline rather than the quality of modern language-model embeddings.

---

# Vector Representation

Vectors are validated before entering the system.

The implementation checks properties including:

* one-dimensional shape,
* non-empty vectors,
* finite numeric values,
* consistent dimensions,
* valid IDs.

Vectors are represented internally using NumPy arrays.

Cosine similarity is used as the similarity metric.

For non-zero vectors:

```text
cosine_similarity(a, b)
    = (a · b) / (||a|| ||b||)
```

Zero vectors are handled explicitly by the implementation.

---

# Exact Search

The exact index performs brute-force cosine similarity against all active vectors.

Conceptually:

```text
Query
  |
  v
Compare with every stored vector
  |
  v
Calculate cosine similarity
  |
  v
Sort by similarity
  |
  v
Return top-k
```

This is computationally expensive as the dataset grows, but it provides an exact answer.

Therefore, the brute-force index is used as the **ground-truth oracle** for evaluating IVF-Flat.

---

# IVF-Flat

The approximate index implemented in this project is **IVF-Flat**.

IVF stands for Inverted File.

The implementation consists of:

1. K-means clustering
2. Cluster centroids
3. Inverted lists
4. Query-to-centroid assignment
5. Candidate collection
6. Exact cosine similarity over candidates
7. Top-k ranking

---

## Build Process

Given the dataset:

```text
50,000 vectors
        |
        v
    K-means
        |
        v
100 cluster centroids
        |
        v
Assign every vector to its nearest centroid
        |
        v
Build inverted lists
```

Each inverted list contains the IDs of vectors assigned to a particular cluster.

---

## Search Process

For a query:

```text
Query
  |
  v
Compare query against all centroids
  |
  v
Select n_probe closest clusters
  |
  v
Collect vectors from those clusters
  |
  v
Calculate exact cosine similarity
  |
  v
Return top-k
```

Unlike the brute-force index, IVF-Flat does **not** compare the query with every vector.

It first narrows the search to a candidate subset.

---

# API

The project exposes a common interface:

```python
insert(id, vector)
search(query, k)
delete(id)
```

Example:

```python
from vector_db.api.index import VectorIndex

index = VectorIndex(backend)

index.insert(1, vector)

results = index.search(query, k=10)

index.delete(1)
```

The common API allows different indexing implementations to be used through the same interface.

---

# Deletion

The exact index supports deletion without physically shifting all vectors.

Deleted rows become inactive and can later be reused.

This provides tombstone-like behavior while avoiding expensive array compaction after every deletion.

The IVF implementation removes deleted IDs from its inverted-list membership and stored-vector mappings.

---

# Evaluation Methodology

The approximate index is evaluated against the exact brute-force index.

For each query:

```text
                 Query
                   |
          +--------+--------+
          |                 |
          v                 v
   Brute Force           IVF-Flat
     exact                 approx.
          |                 |
          v                 v
      Ground Truth       Approx. Top-k
          |                 |
          +--------+--------+
                   |
                   v
              Recall@k
```

The benchmark measures:

* Recall@1
* Recall@5
* Recall@10
* P50 latency
* P95 latency
* P99 latency
* minimum latency
* maximum latency
* mean latency
* candidate count
* candidate fraction
* index build time

---

# Final Benchmark

The verified baseline benchmark used:

```text
Vectors:             50,000
Queries:             500
Dimension:           128
Clusters:            100
Probes:              8
K-Means iterations:  10
```

### Build

```text
Build time: 1.036043 s
```

### Recall

```text
Recall@1:  1.0000
Recall@5:  1.0000
Recall@10: 1.0000
```

### Latency

|  k |       P50 |       P95 |       P99 |      Mean |
| -: | --------: | --------: | --------: | --------: |
|  1 | 0.0451 ms | 0.1120 ms | 0.1757 ms | 0.0534 ms |
|  5 | 0.0443 ms | 0.1135 ms | 0.1837 ms | 0.0536 ms |
| 10 | 0.0443 ms | 0.1242 ms | 0.1846 ms | 0.0545 ms |

### Candidate reduction

```text
Mean candidate count:   4,057.80
Mean candidate fraction: 8.12%
```

Therefore, with 100 clusters and 8 probes, IVF-Flat examined approximately **8.12% of the dataset on average** while achieving 100% recall on this synthetic benchmark.

> These latency measurements are benchmark measurements from this implementation and environment. They should not be interpreted as production performance guarantees.

---

# `n_probe` Experiment

The `n_probe` parameter controls how many clusters are searched for each query.

Higher `n_probe` generally means:

```text
More clusters
    ->
More candidate vectors
    ->
More computation
    ->
Potentially higher recall
```

Verified results with 100 clusters:

| n_probe |  Recall@10 |       P95 | Candidates | Candidate fraction |
| ------: | ---------: | --------: | ---------: | -----------------: |
|       1 |     0.9726 |   9.92 ms |     575.69 |              1.15% |
|       2 | **1.0000** |  12.60 ms |   1,066.27 |              2.13% |
|       4 | **1.0000** |  20.82 ms |   2,072.99 |              4.15% |
|       8 | **1.0000** |  40.44 ms |   4,057.80 |              8.12% |
|      16 | **1.0000** |  76.64 ms |   8,066.23 |             16.13% |
|      32 | **1.0000** | 146.93 ms |  16,050.88 |             32.10% |
|      64 | **1.0000** | 563.20 ms |  31,958.61 |             63.92% |
|     100 | **1.0000** | 628.28 ms |  50,000.00 |            100.00% |

### Observation

`n_probe=1` achieved 97.26% Recall@10 while examining only 1.15% of the dataset.

Increasing it to `n_probe=2` increased Recall@10 to 100% while examining only 2.13% of the dataset.

Beyond `n_probe=2`, recall remained at 100% on this benchmark while search cost continued to increase.

This makes `n_probe=2` an attractive setting for this particular dataset.

---

# `n_clusters` Experiment

The number of clusters controls how finely the dataset is partitioned.

Verified results with `n_probe=8`:

| n_clusters | Recall@1 | Recall@5 | Recall@10 | Build time |          P95 | Candidates |
| ---------: | -------: | -------: | --------: | ---------: | -----------: | ---------: |
|         50 |   1.0000 |   1.0000 |    1.0000 |   0.9245 s |     83.62 ms |   8,064.71 |
|        100 |   1.0000 |   1.0000 |    1.0000 |   0.7384 s |     37.66 ms |   4,057.80 |
|        200 |   1.0000 |   1.0000 |    1.0000 |   1.0986 s | **19.02 ms** |   2,083.81 |

Candidate fractions:

```text
50 clusters   → 16.13%
100 clusters  →  8.12%
200 clusters  →  4.17%
```

### Observation

Increasing the number of clusters reduced the number of candidate vectors and improved measured search latency in these experiments.

The 200-cluster configuration had the lowest measured P95 latency and candidate fraction.

However, it also had the highest measured build time of the three configurations.

The experiment therefore demonstrates the trade-off between index granularity, build cost, and search cost.

---

# Recommended IVF Settings

Based on the tested combinations, a reasonable project default is:

```yaml
ivf:
  n_clusters: 100
  n_probe: 2
  kmeans_iterations: 10
```

Why:

* 100 clusters provides a good middle ground in the tested `n_clusters` experiment.
* `n_probe=2` achieved 100% Recall@10 in the tested benchmark.
* It examined only about 2.13% of the dataset.
* Increasing `n_probe` beyond 2 did not improve recall on this dataset but increased measured search cost.

However, **100 clusters + 2 probes was not directly tested as a combined configuration in the `n_clusters` experiment**. The recommendation is therefore based on the two parameter sweeps rather than a full grid search.

For a more conservative configuration, the existing benchmark setting remains:

```yaml
n_clusters: 100
n_probe: 8
```

This configuration was directly benchmarked and achieved 100% recall at all tested k values.

---

# Memory Measurement

The project includes:

```powershell
python scripts\measure_memory.py
```

The script uses process RSS measurement through `psutil`.

A verified run on Windows reported approximately:

```text
Process RSS:       80.78 MB
Raw dataset:       48.83 MB
Brute-force increase: 54.84 MB
IVF-Flat increase:   121.52 MB
```

IVF-Flat requires additional memory for its index structures, including stored vectors, centroids, and inverted-list/mapping structures.

RSS measurements are approximate process-level measurements rather than exact Python object memory accounting.

---

# Semantic Search Demo

Run:

```powershell
python scripts\semantic_search_demo.py
```

The demo accepts text and converts it into a deterministic toy embedding before searching the custom vector database.

Example pipeline:

```text
"I love coding software"
        |
        v
Toy text embedding
        |
        v
IVF-Flat
        |
        v
Similarity search
        |
        v
"I like developing software and writing code."
```

The demo is intended to show how a vector database can sit underneath a semantic-search application.

### Important limitation

The text embedding function is intentionally simple.

It is **not** equivalent to embeddings generated by models such as modern transformer-based embedding models.

Consequently, semantic matches may sometimes be imperfect.

The purpose of this demo is to demonstrate the database/search pipeline, not state-of-the-art semantic understanding.

---

# Running the Tests

Run the complete test suite:

```powershell
pytest -v
```

The verified full test run passed:

```text
167 passed
```

Tests cover areas including:

* vector validation
* distance calculations
* brute-force search
* K-means clustering
* inverted lists
* IVF-Flat
* storage
* API behavior
* recall
* latency
* candidate metrics
* benchmarking
* dataset generation
* semantic-search demo

---

# Running the Benchmark

Generate the dataset first if necessary:

```powershell
python scripts\generate_data.py
```

Then run:

```powershell
python scripts\benchmark.py
```

Results are written to:

```text
experiments/results/benchmark_baseline.json
```

The benchmark compares IVF-Flat against brute-force ground truth and reports recall, latency, candidate reduction, and build time.

---

# Running the Experiments

## Probe experiment

```powershell
python scripts\experiment_n_probe.py
```

Results:

```text
experiments/results/n_probe_experiment.json
```

This evaluates the effect of different `n_probe` values.

---

## Cluster experiment

```powershell
python scripts\experiment_n_clusters.py
```

Results:

```text
experiments/results/n_clusters_experiment.json
```

This evaluates the effect of different cluster counts.

---

## Memory experiment

```powershell
python scripts\measure_memory.py
```

Results:

```text
experiments/results/memory_baseline.json
```

---

# Design Decisions

## Why brute force?

Brute force is simple, exact, and easy to reason about.

It serves as the correctness oracle for approximate search.

## Why IVF-Flat?

IVF-Flat provides a clear introduction to approximate nearest-neighbor search without requiring a complex graph implementation.

It exposes important concepts such as:

* clustering,
* partitions,
* inverted lists,
* candidate generation,
* recall/cost trade-offs.

## Why synthetic data?

The dataset is deterministic and intentionally clustered so that IVF behavior can be measured reproducibly.

This avoids relying on external embedding services or large external datasets.

## Why NumPy?

NumPy provides efficient numerical operations while keeping the actual vector-search logic visible.

The project does not delegate nearest-neighbor indexing to a specialized ANN library.

---

# Constraints

This project is intentionally built from first principles.

The implementation does **not** use:

* FAISS
* Pinecone
* Chroma
* sklearn nearest-neighbor implementations
* other external ANN/vector-index implementations

The goal is to implement the important indexing and search logic inside this repository.

---

# Limitations

This is an educational vector database rather than a production database.

Current limitations include:

* in-memory storage
* no persistent vector storage
* no distributed architecture
* no replication
* no sharding
* no authentication
* no network service/API server
* no production-grade embedding model
* synthetic benchmark dataset
* limited index configuration search
* no concurrent query/update architecture
* no advanced quantization
* no HNSW implementation
* no production durability guarantees

The benchmark results should therefore be interpreted as measurements of this implementation under the specified synthetic workload.

---

# What This Project Demonstrates

The project demonstrates the complete basic vector-search pipeline:

```text
Data
 ->
Vector validation
 ->
Storage
 ->
Index construction
 ->
K-means clustering
 ->
Inverted lists
 ->
Query
 ->
Cluster selection
 ->
Candidate generation
 ->
Cosine similarity
 ->
Top-k results
 ->
Recall / latency evaluation
```

More importantly, the experiments demonstrate that approximate vector search is fundamentally a **trade-off**.

Searching fewer candidates can reduce computation but may reduce recall.

Searching more candidates can improve recall but increases search cost.

IVF-Flat exposes this trade-off through:

```text
n_clusters
    +
n_probe
```

---

# Summary of Verified Results

For the verified baseline:

```text
Dataset:
  50,000 vectors
  500 queries
  128 dimensions

IVF:
  100 clusters
  8 probes

Recall:
  Recall@1  = 100%
  Recall@5  = 100%
  Recall@10 = 100%

Candidate fraction:
  8.12%

Build time:
  ~1.04 seconds
```

The parameter experiments showed:

```text
n_probe=1
  → 97.26% Recall@10
  → 1.15% candidates

n_probe=2
  → 100% Recall@10
  → 2.13% candidates
```

and:

```text
50 clusters
  → 16.13% candidates

100 clusters
  → 8.12% candidates

200 clusters
  → 4.17% candidates
```

All tested cluster configurations achieved 100% Recall@1, Recall@5, and Recall@10 with `n_probe=8`.

---

# Conclusion

This project implements a small vector database from first principles and measures the behavior of exact and approximate vector search rather than treating vector databases as a black box.

The main lesson is the relationship between:

```text
Index structure
      ->
Candidate reduction
      ->
Search latency
      ->
Recall
```

The brute-force implementation provides an exact reference, while IVF-Flat demonstrates how clustering can reduce the amount of data examined during search.

The experiments make the trade-offs measurable and reproducible, while the semantic-search demo shows how the underlying vector database can be used in a simple application pipeline.
