# Technical Design — From-Scratch Vector Database

## 1. Design Principles

1. Correctness before speed.
2. Exact search is the oracle.
3. One algorithmic idea per phase.
4. Measure every approximation.
5. Keep algorithms visible.
6. Prefer NumPy operations over unnecessary Python loops.
7. Do not optimize until a benchmark exposes a real bottleneck.

## 2. Vector Representation

Vectors are NumPy arrays with a fixed dimension `D`.

Example:

```text
vector = [x1, x2, ..., xD]
```

The index owns or safely stores the vector representation.

All inserted vectors must have the same dimensionality.

## 3. Cosine Similarity

For vectors `q` and `x`:

```text
similarity(q, x) =
(q · x) / (||q|| * ||x||)
```

For normalized vectors:

```text
similarity(q, x) = q · x
```

### Zero Vectors

A zero vector has undefined cosine similarity.

The implementation must define deterministic behavior. The recommended initial behavior is to treat similarity as zero and document it.

## 4. Exact Brute-Force Index

### State

```text
vectors: matrix [N, D]
ids: array [N]
active: boolean array [N]
```

### Insert

1. Validate dimension.
2. Validate ID.
3. Store vector.
4. Mark active.

### Search

1. Validate query dimension.
2. Normalize query when using cosine similarity.
3. Compare query with all active vectors.
4. Obtain similarity scores.
5. Select top-k.
6. Return sorted results.

### Complexity

For `N` vectors of dimension `D`:

```text
Search: O(ND)
Storage: O(ND)
```

This gives us exact ground truth but scales poorly.

## 5. Top-K Selection

Avoid sorting all `N` results when only `k` are needed.

The first implementation may use NumPy's partial selection facilities.

Important:
- partial selection finds candidates efficiently,
- final selected results should be sorted by score,
- deterministic tie-breaking should be considered.

## 6. IVF-Flat

IVF = Inverted File.

Flat means candidate vectors are not compressed or quantized; their original values are used for exact similarity calculation.

### State

```text
centroids: [C, D]

inverted_lists:
    cluster_0 -> IDs
    cluster_1 -> IDs
    ...
    cluster_C-1 -> IDs

vectors:
    ID -> vector

active:
    ID -> bool
```

Where:
- `C` = number of clusters.

## 7. IVF Build

### Step 1 — Choose Centroids

Use a simple k-means-style procedure.

Possible initialization:
- deterministic random sampling with a seed.

### Step 2 — Assignment

For every vector:

```text
vector → nearest centroid
```

### Step 3 — Update

Recompute each centroid from assigned vectors.

Repeat for a configurable number of iterations.

### Empty Clusters

An empty cluster must be handled explicitly.

Initial strategy:
- reinitialize its centroid from an existing vector,
- record the behavior in tests.

## 8. IVF Search

Inputs:

```text
query
k
n_probe
```

### Step 1

Find the nearest `n_probe` centroids.

### Step 2

Read their inverted lists.

### Step 3

Collect active candidate vectors.

### Step 4

Compute exact cosine similarity against candidates.

### Step 5

Select final top-k.

### Complexity

Approximately:

```text
centroid search: O(CD)
candidate search: O(MD)
```

where `M` is the number of vectors in the selected clusters.

The entire point of IVF is to make:

`M << N`

for appropriate settings.

## 9. IVF Accuracy Tradeoff

If `n_probe` is small:

```text
less work
->
lower latency
->
potentially lower recall
```

If `n_probe` increases:

```text
more work
->
higher latency
->
higher recall
```

At sufficiently high `n_probe`, IVF approaches exact search.

This relationship is the central experiment of the project.

## 10. Insert Into IVF

For an already-built index:

1. validate vector,
2. find nearest centroid,
3. append vector ID to the selected inverted list,
4. store vector,
5. mark active.

Important limitation:

New vectors do not automatically improve centroid quality.

Therefore:
- insertion is supported,
- centroid retraining is a separate operation.

## 11. Delete

Initial design:

```text
delete(id)
    ->
active[id] = false
```

The ID may remain in an inverted list.

Search checks active state before scoring/returning the vector.

### Why Tombstones First?

Physical removal introduces:
- list mutation complexity,
- index-position issues,
- potential expensive compaction.

Tombstones give a correct MVP while exposing a real vector-database problem.

## 12. Update

Do not initially expose `update`.

Treat an update as:

```text
delete(old)
insert(new)
```

unless an explicit update design becomes necessary.

## 13. Evaluation Design

### Ground Truth

For every query:

```text
exact.search(query, 10)
```

is the ground truth.

### Recall

```text
intersection = approximate_ids ∩ exact_ids

recall@10 = len(intersection) / 10
```

Aggregate across 500 queries.

### Candidate Count

For every IVF query:

```text
candidate_count = number of active vectors inspected
```

Report:
- average,
- minimum,
- maximum,
- fraction of total dataset.

### Latency

Warm-up queries should be separated from measured queries where practical.

Measure:
- per-query latency,
- mean,
- p50,
- p95,
- p99.

## 14. Benchmark Matrix

At minimum, sweep:

```text
n_probe = 1
n_probe = 2
n_probe = 4
n_probe = 8
...
```

while holding the dataset and queries constant.

Example result:

```text
n_probe | Recall@10 | Avg Latency | Candidates
--------|-----------|--------------|-----------
1       | ...       | ...          | ...
2       | ...       | ...          | ...
4       | ...       | ...          | ...
8       | ...       | ...          | ...
```

The exact values must come from actual runs.

## 15. Dataset Design

Preferred:

```text
5,000+ short real texts
        ->
embedding model
        ->
vectors
        ->
50,000+ benchmark vectors if the selected source supports expansion
```

However, external embedding dependencies should not be allowed to derail the core project.

Fallback:

```text
cluster centers
      ->
sample vectors around centers
      ->
controlled synthetic embedding-like dataset
```

Use a fixed seed.

## 16. Testing Design

### Unit Tests

- cosine similarity,
- zero vectors,
- dimensions,
- top-k,
- insertion,
- deletion,
- clustering,
- inverted lists,
- IVF search.

### Integration Tests

- build index,
- insert,
- search,
- delete,
- compare exact and IVF.

### Property-Oriented Checks

For random small datasets:

```text
IVF result ⊆ dataset
deleted IDs never returned
k results are returned when enough active vectors exist
exact results match a trusted direct calculation
```

## 17. Error Handling

Reject:
- wrong vector dimension,
- wrong query dimension,
- invalid `k`,
- invalid `n_probe`,
- duplicate IDs where unsupported,
- operations on malformed index state.

Errors should be explicit.

## 18. Persistence — Later Phase

Do not implement persistence in the MVP.

When added, separate:
- logical index state,
- serialized vector data,
- index metadata,
- version/schema information.

Persistence must not be mixed into the search algorithm.

## 19. Advanced Optimization Roadmap

Only after correctness:

### Optimization A
Vector normalization caching.

### Optimization B
More efficient storage layout.

### Optimization C
Batch query support.

### Optimization D
Better centroid initialization.

### Optimization E
Parallel/vectorized candidate scoring.

### Optimization F
Deletion compaction.

### Optimization G
Persistence.

### Optimization H
HNSW.

Each optimization gets its own benchmark comparison.

## 20. Important Design Constraint

Do not hide the algorithm behind a generic abstraction so aggressively that the implementation becomes difficult to understand.

The project is educational.

The code should make it possible to answer:

- Where are vectors stored?
- How is cosine similarity calculated?
- How are centroids selected?
- How is a vector assigned to a cluster?
- How does `n_probe` reduce work?
- How are deleted vectors handled?
- Why did recall decrease?
- Why did latency change?
