# Architecture — From-Scratch Vector Database

## 1. Architectural Goal

Create a small, understandable vector-search system where:

- storage is separated from search algorithms,
- exact search provides ground truth,
- approximate search is independently implemented,
- evaluation can compare both implementations,
- future index types can be added without rewriting the public API.

## 2. High-Level Architecture

```text
                    |---------------------|
                    │      User / CLI     │
                    |----------┬----------|
                               │
                               \/
                    |---------------------|
                    │    Common Index API │
                    │ insert/search/delete│
                    |----------┬----------|
                               │
                 |-------------┴------------|
                 │                           │
                 \/                           \/
       |------------------|        |------------------|
       │ Exact Index      │        │ IVF-Flat Index   │
       │ Brute Force      │        │ Approximate      │
       |--------┬---------|        |--------┬---------|
                │                           │
                \/                           \/
       |------------------|        |------------------|
       │ Vector Storage   │        │ Centroids        │
       │ IDs + vectors    │        │ Inverted Lists   │
       │ active state     │        │ Cluster mapping  │
       |------------------|        |------------------|
                │                           │
                |-------------┬-------------|
                              \/
                    |---------------------|
                    │    Evaluation       │
                    │ recall / latency /  │
                    │ candidates / build  │
                    |---------------------|
```

## 3. Components

### 3.1 Core

Contains:
- vector/type definitions,
- distance functions,
- shared interfaces,
- validation.

It must not know how a particular index is implemented.

### 3.2 Storage

Responsible for:
- IDs,
- vectors,
- active/deleted state,
- dimensionality.

The first implementation can be in-memory.

### 3.3 Exact Index

Brute-force search:

```text
query
  ->
validate dimension
  ->
compare against every active vector
  ->
compute cosine similarity
  ->
select top-k
  ->
return IDs + scores
```

This is the correctness oracle.

### 3.4 IVF-Flat

The index consists of:

```text
Centroids
   │
   ├-- Cluster 0 → vector IDs
   ├-- Cluster 1 → vector IDs
   ├-- Cluster 2 → vector IDs
   |-- ...
```

Build:

```text
vectors
   ->
initialize centroids
   ->
cluster / assign
   ->
create inverted lists
```

Search:

```text
query
  ->
find nearest centroids
  ->
select n_probe clusters
  ->
collect candidate vectors
  ->
exact cosine similarity on candidates
  ->
top-k
```

The critical approximation happens when only `n_probe` clusters are searched.

### 3.5 Evaluation

The evaluator receives:

```text
query
  │
  ├-- Exact Index ------► Ground Truth
  │
  |-- IVF Index --------► Approximate Result
                             │
                             \/
                    Recall + Latency
```

It must not influence search results.

## 4. Package Boundaries

```text
src/vector_db/
├-- core/
├-- exact/
├-- approximate/
│   |-- ivf/
├-- storage/
├-- api/
|-- evaluation/
```

### Dependency Direction

```text
api
 │
 ├-- exact -------► core
 │                  ▲
 |-- approximate --|
          │
          |--------► storage

evaluation ---------► api
```

Algorithm modules should not depend on benchmark/reporting modules.

## 5. Public Interface

Conceptually:

```python
class VectorIndex:
    def insert(self, id, vector): ...
    def search(self, query, k): ...
    def delete(self, id): ...
```

Exact and IVF implementations should satisfy the same behavioral contract.

## 6. Data Model

Each active record has:

```text
VectorRecord
├-- id
├-- vector
|-- active
```

IVF additionally maintains:

```text
IVFIndex
├-- centroids
├-- inverted_lists
|-- assignment metadata
```

## 7. Deletion Strategy

Initial strategy:

- mark vectors deleted,
- retain their physical position,
- ignore them during search.

Why:
- simple,
- safe,
- avoids immediately rebuilding clusters.

Future strategy:
- periodic compaction,
- rebuilding affected inverted lists,
- optional index rebuild.

## 8. Clustering Strategy

The first IVF implementation should use a simple, understandable k-means-style procedure.

Inputs:
- vectors,
- number of clusters,
- number of iterations,
- random seed.

Output:
- centroid matrix,
- vector-to-cluster assignments.

The first version should favor correctness and inspectability over sophisticated initialization.

## 9. Distance Calculation

Primary metric:

Cosine similarity:

`cos(q, x) = (q · x) / (||q|| ||x||)`

For normalized vectors:

`cos(q, x) = q · x`

Normalization should be handled deliberately rather than accidentally.

Zero vectors require explicit behavior.

## 10. Search Result Contract

A result should contain enough information to evaluate behavior:

```text
Result
├-- id
|-- score
```

Ordering:
- highest cosine similarity first.

Ties:
- deterministic tie-breaking should be preferred.

## 11. Benchmark Architecture

```text
Dataset
  │
  ├-- Build exact index
  │
  ├-- Build IVF index
  │
  |-- Generate/load queries
           │
           \/
      Ground Truth
           │
           \/
      IVF Evaluation
           │
     |-----┼---------|
     \/     \/         \/
   Recall Latency Candidates
```

## 12. Integration Points

### Dataset → Index

The dataset must produce vectors with:
- consistent dimensionality,
- deterministic IDs.

### Index → Evaluation

Evaluation must treat indexes as black-box implementations of the common contract.

### Benchmark → Results

Benchmark output should be structured so that experiments can be compared later.

## 13. Scaling Boundaries

The first architecture is intentionally in-memory.

At larger scale, likely pressure points are:

- memory footprint,
- vector copies,
- Python object overhead,
- brute-force scan cost,
- inverted-list size imbalance,
- clustering build time,
- deletion tombstones.

These are expected learning points, not problems to prematurely solve.

## 14. Future Architecture

Possible advanced structure:

```text
                Vector DB
                    │
          |---------┴---------|
          \/                   \/
       Exact              ANN Index
                            │
                 |----------┴----------|
                 \/                     \/
                IVF                   HNSW
```

Persistence, filtering, concurrency, and service APIs should only be added after the core index architecture is stable.
