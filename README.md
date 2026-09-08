# Vector Database From Scratch

A learning-focused vector database implemented from first principles in **Python and NumPy**.

The project demonstrates how modern vector search works internally by implementing the core components of a vector database without relying on specialized vector-search or ANN libraries.

It includes:

* Exact brute-force cosine similarity search
* IVF-Flat approximate nearest-neighbor search
* K-means clustering implemented from scratch
* Inverted lists
* Insert, search, and delete operations
* Recall and latency evaluation
* Candidate-reduction measurements
* Deterministic synthetic benchmarking
* Real semantic-search demonstration using `all-MiniLM-L6-v2`
* Lightweight HTTP API
* Unit, integration, benchmark, API, and semantic-search tests

The goal is not to compete with production vector databases. The goal is to **understand what happens inside one**.

---

## 1. Project Overview

A vector database stores numerical representations of data and retrieves the vectors most similar to a query.

Production systems often hide the underlying indexing and search algorithms behind an API. This project takes the opposite approach: the important pieces are implemented directly so their behavior can be measured and understood.

The project explores the relationship between:

```text
Vectors
   ->
Storage
   ->
Index construction
   ->
Candidate selection
   ->
Similarity computation
   ->
Top-K results
   ->
Recall / latency trade-off
```

The implementation contains two primary search strategies:

### Exact search

A brute-force index compares a query against every active vector.

### Approximate search

An IVF-Flat index first selects promising clusters and then performs exact cosine similarity only over vectors belonging to those clusters.

The brute-force implementation serves as the **correctness oracle** for evaluating IVF-Flat.

---

## 2. Project Goals

This project is intentionally educational.

The main goals are to understand:

* How vector similarity is calculated
* How vectors are stored and validated
* How K-means can partition a vector space
* How inverted lists reduce the search space
* How IVF-Flat performs approximate nearest-neighbor search
* How `n_probe` affects recall and search cost
* How `n_clusters` affects index granularity and candidate reduction
* How approximate search is evaluated against exact ground truth
* How vector search can be used underneath a semantic-search application
* How an indexing layer can be exposed through a simple HTTP API

---

## 3. Key Constraints

The core indexing and search logic is implemented from scratch.

This project does **not** rely on:

* FAISS
* Pinecone
* Chroma
* sklearn nearest-neighbor implementations
* Other external ANN/vector-index implementations

NumPy is used for numerical computation, but the vector-indexing logic remains inside this repository.

---

# 4. Architecture

## High-Level Architecture

```text
                         Query Vector
                              |
                              v
                    +-------------------+
                    |   VectorIndex API |
                    +---------+---------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
       +----------------+          +----------------+
       | BruteForceIndex|          |   IVF-Flat     |
       |  Exact Search  |          | Approx. Search |
       +-------+--------+          +-------+--------+
               |                           |
               |                           v
               |                   +---------------+
               |                   |   K-means     |
               |                   |   Centroids   |
               |                   +-------+-------+
               |                           |
               |                           v
               |                   +---------------+
               |                   | Inverted Lists|
               |                   +-------+-------+
               |                           |
               |                           v
               |                    Candidate Vectors
               |                           |
               +-------------+-------------+
                             |
                             v
                    Cosine Similarity
                             |
                             v
                        Top-K Results
```

## Search Flow

```text
Query Vector
     ->
Find nearest centroids
     ->
Probe top N clusters
     ->
Collect candidate vectors
     ->
Compute exact cosine similarity
     ->
Apply deterministic ranking
     ->
Return Top-K results
```

---

vector-database/
|
|--- configs/
|   |--- default.yaml
|
|--- docs/
|   |--- Constraints.md
|   |--- Decisions.md
|   |--- Flow.md
|   |--- Handover.md
|   |--- ROLLBACK.md
|   |--- TEST_CHECKLIST.md
|   |--- features/
|       |--- FEATURE-vector-index.md
|
|--- experiments/
|   |--- notebooks/
|   |--- results/
|
|--- frontend/
|   |--- app.js
|   |--- index.html
|   |--- README.md
|   |--- style.css
|
|--- scripts/
|   |--- benchmark.py
|   |--- build_index.py
|   |--- experiment_n_clusters.py
|   |--- experiment_n_probe.py
|   |--- generate_data.py
|   |--- measure_memory.py
|   |--- semantic_search_demo.py
|   |--- serve_frontend.py
|
|--- src/
|   |--- vector_db/
|       |--- api/
|       |--- approximate/
|       |   |--- ivf/
|       |--- core/
|       |--- evaluation/
|       |--- exact/
|       |--- storage/
|
|--- tests/
|   |--- benchmarks/
|   |--- integration/
|   |--- unit/
|
|--- architecture.md
|--- design.md
|--- phases.md
|--- PRD.md
|--- pyproject.toml
|--- README.md
---

# 6. Exact Brute-Force Search

The exact index performs brute-force cosine similarity against every active vector.

Conceptually:

```text
Query
  ->
Compare with every stored vector
  ->
Calculate cosine similarity
  ->
Rank results
  ->
Return Top-K
```

For non-zero vectors:

```text
cosine_similarity(a, b)
    = (a · b) / (||a|| ||b||)
```

The implementation explicitly handles zero vectors and validates vectors before they enter the system.

### Why brute force?

Brute-force search is:

* Simple
* Exact
* Easy to reason about
* Independent of approximate-index behavior

Therefore, it provides the **ground-truth result set** used to evaluate IVF-Flat.

The ranking also uses deterministic tie-breaking so repeated runs produce stable results.

---

# 7. IVF-Flat

The approximate index implemented in this project is **IVF-Flat**.

IVF stands for **Inverted File**.

The implementation consists of:

1. K-means clustering
2. Cluster centroids
3. Inverted lists
4. Query-to-centroid assignment
5. Candidate selection
6. Exact cosine similarity over candidates
7. Top-K ranking

Unlike brute-force search, IVF-Flat does not compare a query against every stored vector.

Instead, it narrows the search space first.

---

## 7.1 Index Construction

Given the dataset:

```text
50,000 vectors
      ->
   K-means
      ->
100 cluster centroids
      ->
Assign each vector to nearest centroid
      ->
Build inverted lists
```

Each inverted list stores the IDs of vectors assigned to a particular cluster.

---

## 7.2 Query Processing

For every query:

```text
Query
  ->
Compare query with cluster centroids
  ->
Select nearest n_probe clusters
  ->
Collect candidate vector IDs
  ->
Compute exact cosine similarity
  ->
Rank candidates
  ->
Return Top-K
```

This produces an important trade-off:

```text
More probes
    ->
More candidates
    ->
More computation
    ->
Potentially higher recall
```

Conversely:

```text
Fewer probes
    ->
Fewer candidates
    ->
Lower search cost
    ->
Potentially lower recall
```

---

# 8. Vector Representation and Validation

Vectors are represented internally using NumPy arrays.

Before insertion, vectors are validated for properties including:

* One-dimensional shape
* Non-empty dimensions
* Finite numeric values
* Consistent dimensionality
* Valid IDs

Cosine similarity is used as the distance/similarity metric.

Zero vectors are handled explicitly rather than allowing undefined normalization behavior.

---

# 9. Common API

The indexing implementations expose a common interface:

```python
insert(id, vector)
search(query, k)
delete(id)
```

This allows different index implementations to be used through a consistent API.

Example:

```python
from vector_db.api.index import VectorIndex

index = VectorIndex(backend)

index.insert(1, vector)

results = index.search(query, k=10)

index.delete(1)
```

The same conceptual operations can therefore be performed against exact and approximate backends.

---

# 10. Insert and Delete

## Insert

Vectors are validated before being stored and indexed.

The IVF implementation assigns inserted vectors to the appropriate cluster and updates its inverted-list structures.

## Delete

The exact index uses inactive/free rows rather than physically shifting all subsequent vectors after every deletion.

This provides tombstone-like behavior while allowing deleted storage slots to be reused.

The IVF implementation removes deleted IDs from its inverted-list membership and associated mappings.

---

# 11. Benchmark Dataset

The primary benchmark uses a **deterministic clustered synthetic dataset**.

### Dataset

| Parameter    |     Value |
| ------------ | --------: |
| Vectors      |    50,000 |
| Queries      |       500 |
| Dimensions   |       128 |
| Random seed  |        42 |
| Source       | Synthetic |
| Distribution | Clustered |
| Normalized   |       Yes |

The data generator creates normalized cluster centers and generates vectors around those centers using controlled Gaussian noise.

Generate the dataset with:

```powershell
python scripts\generate_data.py
```

The generated dataset is written to:

```text
data/processed/dataset.npz
```

Generated benchmark data is not treated as source code and is ignored by Git.

---

# 12. Synthetic vs Real Data

It is important to distinguish the two workloads in this project.

## Synthetic benchmark

The benchmark uses:

* 50,000 synthetic vectors
* 500 queries
* 128 dimensions
* Deterministic clustered distribution
* Seed `42`

This dataset is designed for controlled and reproducible experiments.

It should **not** be interpreted as representative of every real-world embedding workload.

## Semantic-search demo

The semantic-search demo uses:

* Real natural-language statements
* A pretrained embedding model
* `all-MiniLM-L6-v2`
* 384-dimensional embeddings
* Normalized sentence embeddings
* The project's own IVF-Flat implementation for search

The semantic demo therefore uses a realistic embedding model while keeping the vector database/index implementation entirely within this project.

---

# 13. Benchmark Results

## Baseline Configuration

The verified benchmark configuration is:

| Parameter          |  Value |
| ------------------ | -----: |
| Vectors            | 50,000 |
| Queries            |    500 |
| Dimensions         |    128 |
| Clusters           |    100 |
| Probes             |      2 |
| K-means iterations |     10 |
| Dataset seed       |     42 |
| IVF seed           |     42 |

## Results

| Metric                  |       Result |
| ----------------------- | -----------: |
| Recall@1                |     **100%** |
| Recall@5                |     **100%** |
| Recall@10               |     **100%** |
| Mean candidates         | **1,066.27** |
| Mean candidate fraction |    **2.13%** |

These results come from the deterministic clustered synthetic benchmark dataset.

The configuration achieved perfect recall on the tested workload while examining only about **2.13% of the dataset on average**.

### Latency

Latency should be interpreted as a measurement of this implementation on the benchmark environment rather than a production performance guarantee.

When reporting benchmark latency, the project distinguishes search latency from the total benchmark runtime, which also includes computing exact brute-force ground truth.

> Benchmark measurements depend on hardware, Python/NumPy versions, operating-system scheduling, and runtime conditions.

---

# 14. `n_probe` Experiment

The `n_probe` parameter controls how many clusters are searched for each query.

Verified results with 100 clusters:

| `n_probe` | Recall@10 |       P95 | Candidates | Candidate Fraction |
| --------: | --------: | --------: | ---------: | -----------------: |
|         1 |    97.26% |   9.92 ms |     575.69 |              1.15% |
|         2 |  **100%** |  12.60 ms |   1,066.27 |              2.13% |
|         4 |  **100%** |  20.82 ms |   2,072.99 |              4.15% |
|         8 |  **100%** |  40.44 ms |   4,057.80 |              8.12% |
|        16 |  **100%** |  76.64 ms |   8,066.23 |             16.13% |
|        32 |  **100%** | 146.93 ms |  16,050.88 |             32.10% |
|        64 |  **100%** | 563.20 ms |  31,958.61 |             63.92% |
|       100 |  **100%** | 628.28 ms |  50,000.00 |            100.00% |

### Observation

`n_probe=1` reduced the search space dramatically but achieved only **97.26% Recall@10**.

Increasing to `n_probe=2` produced **100% Recall@10** while examining only **2.13%** of the dataset.

Beyond `n_probe=2`, recall remained at 100% on this particular dataset while candidate count and measured search cost continued to increase.

Therefore, `n_probe=2` is an attractive configuration for this specific benchmark.

This should not be interpreted as a universal optimal value.

---

# 15. `n_clusters` Experiment

The number of clusters controls how finely the vector space is partitioned.

Verified results with `n_probe=8`:

| `n_clusters` | Recall@1 | Recall@5 | Recall@10 | Build Time |          P95 | Candidates |
| -----------: | -------: | -------: | --------: | ---------: | -----------: | ---------: |
|           50 |     100% |     100% |      100% |   0.9245 s |     83.62 ms |   8,064.71 |
|          100 |     100% |     100% |      100% |   0.7384 s |     37.66 ms |   4,057.80 |
|          200 |     100% |     100% |      100% |   1.0986 s | **19.02 ms** |   2,083.81 |

Candidate fractions:

```text
50 clusters
    → 16.13%

100 clusters
    →  8.12%

200 clusters
    →  4.17%
```

### Observation

Increasing the number of clusters reduced candidate counts and improved measured search latency in these experiments.

The 200-cluster configuration produced the lowest measured P95 latency and candidate fraction, but also had the highest measured build time of the three configurations.

This demonstrates the trade-off between:

```text
Number of clusters
        ->
Index granularity
        ->
Candidate count
        ->
Search cost
```

while also increasing index-construction work.

---

# 16. Recommended IVF Configuration

Based on the tested parameter sweeps, the project can use:

```yaml
ivf:
  n_clusters: 100
  n_probe: 2
  kmeans_iterations: 10
```

Why?

* 100 clusters provide a middle ground in the tested cluster-count experiment.
* `n_probe=2` achieved 100% Recall@10 in the tested probe experiment.
* It examined only approximately 2.13% of the dataset.
* Increasing `n_probe` beyond 2 did not improve recall on this dataset.

### Important qualification

The exact combination of:

```text
n_clusters=100
n_probe=2
```

was evaluated through the `n_probe` sweep, while the cluster experiment independently used `n_probe=8`.

Therefore, the recommendation is based on the parameter experiments rather than a complete `n_clusters × n_probe` grid search.

For a directly verified conservative baseline, use:

```yaml
n_clusters: 100
n_probe: 8
```

---

# 17. Semantic Search Demo

The project includes a small semantic-search application demonstrating how an embedding model can be connected to the custom vector database.

## Model

The demo uses:

```text
all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

The model produces normalized sentence embeddings that are then inserted into the project's own vector index.

## Pipeline

```text
User Statement
      ->
all-MiniLM-L6-v2
      ->
384-D embedding
      ->
Own IVF-Flat index
      ->
Similarity search
      ->
Best matching statements
```

## Run

```powershell
python scripts\semantic_search_demo.py
```

Example:

```text
Query:
I like writing software and solving coding problems.

Best matches
------------

1. I like developing software and writing code.
2. I enjoy programming and building software applications.
...
```

The embedding model provides the semantic representation.

The actual vector storage, indexing, candidate selection, similarity search, and ranking are performed by this project.

### Important limitation

The semantic demo is intentionally small and uses a limited collection of statements.

It demonstrates the **vector-search pipeline**, not state-of-the-art semantic-search quality.

The semantic demo should also be considered separately from the synthetic benchmark: benchmark recall and latency results are measured on the controlled synthetic dataset, not on the semantic-demo statements.

---

# 18. HTTP API

The project also exposes the vector index through a lightweight HTTP API.

## Start the server

```powershell
uvicorn vector_db.api.server:app --reload
```

The interactive API documentation is available through FastAPI's generated documentation.

## Endpoints

| Method   | Endpoint       | Purpose                       |
| -------- | -------------- | ----------------------------- |
| `GET`    | `/health`      | Health check                  |
| `POST`   | `/insert`      | Insert a vector               |
| `POST`   | `/search`      | Search for nearest vectors    |
| `DELETE` | `/delete/{id}` | Delete a vector               |
| `GET`    | `/docs`        | Interactive API documentation |

### Insert

```http
POST /insert
Content-Type: application/json
```

Example request:

```json
{
  "id": 50001,
  "vector": [0.0, 0.0, "... 128 values ..."]
}
```

### Search

```http
POST /search
Content-Type: application/json
```

Example request:

```json
{
  "query": [0.0, 0.0, "... 128 values ..."],
  "k": 5
}
```

Example response:

```json
{
  "results": [
    {
      "id": 1,
      "score": 1.0
    }
  ]
}
```

### Delete

```http
DELETE /delete/1
```

The HTTP layer provides a simple interface around the underlying vector-index functionality.

---

# 19. Frontend Demo

The repository includes a lightweight HTML/CSS/JavaScript frontend that connects to the
existing FastAPI server.

The frontend is intentionally a thin client. It does not reimplement IVF-Flat, K-means,
cosine similarity, or vector storage in JavaScript.

## Start the backend

From the repository root:

```powershell
python -m uvicorn vector_db.api.server:app --host 127.0.0.1 --port 8000

# 20. Installation

## Requirements

* Python 3.10+
* NumPy
* PyYAML
* pytest
* psutil
* FastAPI / Uvicorn for the HTTP API
* `sentence-transformers` for the semantic-search demo

## Clone

```powershell
git clone https://github.com/Mayank1705-web/vector-database.git
cd vector-database
```

## Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

## Install the project

```powershell
python -m pip install -e ".[dev]"
```

If `psutil` is not installed:

```powershell
python -m pip install psutil
```

---

# 21. Configuration

The default configuration is stored in:

```text
configs/default.yaml
```

Example:

```yaml
dataset:
  n_vectors: 50000
  n_queries: 500
  dim: 128
  seed: 42
  source: synthetic

ivf:
  n_clusters: 100
  n_probe: 2
  kmeans_iterations: 10

evaluation:
  k_values: [1, 5, 10]
  latency_percentiles: [50, 95, 99]
```

The experiments independently vary `n_probe` and `n_clusters`.

---

# 22. Reproducibility

The project is designed so that the main benchmark can be regenerated from a clean checkout.

## 1. Install dependencies

```powershell
python -m pip install -e ".[dev]"
```

## 2. Generate the dataset

```powershell
python scripts\generate_data.py
```

The benchmark uses deterministic seed:

```text
42
```

## 3. Run the benchmark

```powershell
python scripts\benchmark.py
```

Results are written to:

```text
experiments/results/benchmark_baseline.json
```

## 4. Run tests

```powershell
pytest -v
```

### Reproducibility notes

The benchmark configuration is defined in:

```text
configs/default.yaml
```

Important parameters include:

```text
Dataset size:       50,000
Queries:            500
Dimensions:         128
Dataset seed:       42
IVF seed:           42
Clusters:           100
Probes:             2
K-means iterations: 10
```

Exact latency measurements can vary across machines and runtime environments even when the dataset and random seeds are identical.

---

# 23. Running Experiments

## `n_probe` experiment

```powershell
python scripts\experiment_n_probe.py
```

Results:

```text
experiments/results/n_probe_experiment.json
```

This evaluates how the number of probed clusters affects:

* Recall
* Candidate count
* Candidate fraction
* Search latency

## `n_clusters` experiment

```powershell
python scripts\experiment_n_clusters.py
```

Results:

```text
experiments/results/n_clusters_experiment.json
```

This evaluates the effect of index granularity on:

* Recall
* Candidate count
* Search latency
* Build time

---

# 24. Memory Measurement

The project includes a process-level memory measurement script:

```powershell
python scripts\measure_memory.py
```

The script uses `psutil` to measure process RSS.

A verified Windows run reported approximately:

```text
Process RSS:          80.78 MB
Raw dataset:          48.83 MB
Brute-force increase: 54.84 MB
IVF-Flat increase:   121.52 MB
```

IVF-Flat requires additional memory for structures such as:

* Stored vectors
* Cluster centroids
* Inverted lists
* ID mappings

RSS is a process-level measurement and should not be interpreted as exact Python-object memory accounting.

Results can vary between environments.

---

# 25. Testing

Run the complete test suite:

```powershell
pytest -v
```

Final verified test result:

```text
173 passed
```

The test suite covers areas including:

* Vector validation
* Similarity calculations
* Brute-force search
* Deterministic ranking
* K-means clustering
* Inverted lists
* IVF-Flat
* Storage
* Insert/delete behavior
* Common API behavior
* Recall evaluation
* Latency evaluation
* Candidate metrics
* Dataset generation
* Benchmark functionality
* HTTP API behavior
* Semantic-search demo

---

# 26. Design Decisions

## Why brute force?

Brute force provides an exact and transparent reference implementation.

It makes it possible to measure approximate-search recall without depending on another ANN implementation.

## Why IVF-Flat?

IVF-Flat is a useful introduction to approximate nearest-neighbor indexing because its internal behavior is easy to observe:

```text
Clustering
   ->
Partitioning
   ->
Candidate selection
   ->
Exact similarity
```

It demonstrates the fundamental recall-versus-computation trade-off without introducing the additional complexity of graph-based indexes such as HNSW.

## Why NumPy?

NumPy provides efficient vectorized numerical operations while keeping the indexing and search algorithms visible.

## Why deterministic synthetic data?

A deterministic clustered dataset provides:

* Reproducibility
* Controlled experiments
* Stable comparisons
* Easy regeneration
* No dependency on external datasets or embedding APIs

## Why a pretrained embedding model for the semantic demo?

A real embedding model makes the semantic-search demonstration more meaningful than a hand-written toy embedding function.

`all-MiniLM-L6-v2` converts natural-language statements into semantic vector representations, while the project's own IVF-Flat implementation performs the subsequent vector search.

---

# 27. Limitations

This is an **educational vector database**, not a production database.

Current limitations include:

* In-memory storage
* No persistence layer
* No crash recovery or durability guarantees
* No replication
* No distributed architecture
* No sharding
* No concurrent query/update architecture
* K-means build cost increases with dataset size
* IVF-Flat recall depends on index configuration
* Limited parameter-search coverage
* No advanced quantization
* No HNSW implementation
* No production-scale benchmarking
* Synthetic benchmark rather than a large real-world embedding corpus
* Small statement collection for the semantic demo
* No production-grade authentication or security model
* HTTP API is intended as a lightweight demonstration rather than a production service

---

# 28. What This Project Demonstrates

The project implements the basic vector-search pipeline end-to-end:

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
Top-K ranking
  ->
Recall / latency evaluation
```

The most important lesson is that approximate vector search is a **trade-off**.

Searching fewer candidates:

```text
-> Search cost
-> Candidate count
Potentially -> Recall
```

Searching more candidates:

```text
↑ Search cost
↑ Candidate count
Potentially ↑ Recall
```

IVF-Flat exposes this trade-off through two important parameters:

```text
n_clusters
     +
n_probe
```

The experiments make these relationships measurable instead of treating vector search as a black box.

---

# 29. Summary of Verified Results

For the primary benchmark configuration:

```text
Dataset
  50,000 vectors
  500 queries
  128 dimensions
  Seed = 42

IVF-Flat
  100 clusters
  2 probes
  10 K-means iterations

Recall
  Recall@1  = 100%
  Recall@5  = 100%
  Recall@10 = 100%

Candidates
  Mean candidates        = 1,066.27
  Mean candidate fraction = 2.13%
```

The `n_probe` experiment showed:

```text
n_probe = 1
  → 97.26% Recall@10
  → 1.15% candidates

n_probe = 2
  → 100% Recall@10
  → 2.13% candidates

n_probe > 2
  → 100% Recall@10 on this dataset
  → Increasing candidate count and measured search cost
```

The `n_clusters` experiment showed:

```text
50 clusters
  → 16.13% candidates

100 clusters
  → 8.12% candidates

200 clusters
  → 4.17% candidates
```

All tested cluster configurations achieved:

```text
Recall@1  = 100%
Recall@5  = 100%
Recall@10 = 100%
```

when evaluated with `n_probe=8`.

---

# 30. Conclusion

This project builds a small vector database from first principles and exposes the mechanisms that are normally hidden inside production vector-search systems.

It combines:

* Exact brute-force search
* IVF-Flat approximate indexing
* K-means clustering
* Inverted lists
* Insert/search/delete operations
* Recall and latency evaluation
* Candidate-reduction analysis
* Reproducible benchmarking
* Real semantic embeddings
* A lightweight HTTP API

The central idea is simple:

```text
Index Structure
      ->
Candidate Reduction
      ->
Search Cost
      ->
Recall
```

Rather than treating vector databases as black boxes, this project makes the underlying algorithms, trade-offs, and performance characteristics visible and measurable.

> **A vector database built from scratch to understand how vector search actually works.**
