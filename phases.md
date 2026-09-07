# Development Phases — From Basic to Advanced

## Phase 0 — Repository Foundation

### Objective
Create the project skeleton and development workflow.

### Deliverables
- package structure,
- test structure,
- benchmark structure,
- documentation,
- dependency configuration,
- reproducibility configuration.

### Exit Criteria
The project installs/runs and a minimal test executes.

---

## Phase 1 — Vectors and Distance

### Objective
Understand vector representation and cosine similarity.

### Build
- vector validation,
- cosine similarity,
- batch similarity using NumPy,
- zero-vector handling.

### Tests
- identical vectors,
- orthogonal vectors,
- opposite vectors,
- normalized vectors,
- zero vectors,
- dimension mismatch.

### Exit Criteria
Distance calculations are independently verified.

---

## Phase 2 — Exact Brute Force

### Objective
Build the ground-truth vector index.

### Build
- vector storage,
- stable IDs,
- insert,
- search,
- delete,
- top-k selection.

### Benchmark
Start with:

```text
1,000 vectors
10,000 vectors
50,000 vectors
```

### Exit Criteria
Exact search returns correct top-k results.

---

## Phase 3 — Common Index API

### Objective
Make different index implementations interchangeable.

### API

```text
insert(id, vector)
search(query, k)
delete(id)
```

### Exit Criteria
The exact index can be used entirely through the common API.

---

## Phase 4 — Dataset and Ground Truth

### Objective
Create the real benchmark foundation.

### Requirements
- 50,000+ vectors,
- 500 queries,
- exact top-10 ground truth,
- fixed random seed/configuration.

### Preferred Data
Real text embeddings.

### Fallback
Clustered synthetic vectors.

### Exit Criteria
Ground truth can be regenerated deterministically.

---

## Phase 5 — K-Means / Centroid Builder

### Objective
Build the clustering mechanism required by IVF.

### Build
- centroid initialization,
- vector assignment,
- centroid recomputation,
- configurable iterations,
- empty-cluster handling.

### Tests
Use tiny datasets where cluster assignments can be inspected manually.

### Exit Criteria
Centroids and assignments are stable and testable.

---

## Phase 6 — IVF Inverted Lists

### Objective
Turn clustering into an index structure.

### Build

```text
vector
  ->
nearest centroid
  ->
cluster ID
  ->
inverted list
```

### Exit Criteria
Every active vector belongs to exactly one list.

---

## Phase 7 — IVF-Flat Search

### Objective
Implement approximate nearest-neighbor search.

### Search

```text
query
 ->
nearest centroids
 ->
n_probe
 ->
candidate vectors
 ->
exact cosine scoring
 ->
top-k
```

### Exit Criteria
Search returns valid results and never returns deleted vectors.

---

## Phase 8 — Approximation Measurement

### Objective
Measure exactly what approximation costs.

### Compare
- exact recall,
- IVF recall,
- latency,
- candidate count.

### Sweep
Multiple `n_probe` values.

### Key Output

A recall-vs-latency curve.

### Exit Criteria
The project can clearly explain the speed/accuracy tradeoff.

---

## Phase 9 — API Completeness

### Objective
Make the index usable as a small database component.

### Add
- robust validation,
- duplicate-ID policy,
- deletion semantics,
- deterministic ordering,
- clear errors.

### Exit Criteria
Exact and IVF indexes satisfy the same documented behavior.

---

## Phase 10 — Performance Engineering

### Objective
Optimize only after measurements expose bottlenecks.

### Investigate
- vector normalization,
- memory layout,
- candidate scoring,
- top-k selection,
- clustering,
- batch queries.

### Rule
One optimization per experiment.

### Exit Criteria
Every optimization has before/after measurements.

---

## Phase 11 — Persistence

### Objective
Allow the index to survive process restarts.

### Potential State
- vectors,
- IDs,
- active flags,
- centroids,
- inverted lists,
- configuration/version.

### Risks
- schema compatibility,
- partial writes,
- corruption,
- stale index metadata.

### Exit Criteria
An index can be saved and loaded with identical search behavior.

---

## Phase 12 — Deletion and Compaction

### Objective
Solve the problem intentionally deferred by the MVP.

### Build
- tombstone tracking,
- compaction,
- list rebuilding,
- optional full rebuild.

### Measure
- memory before/after,
- deletion throughput,
- search latency before/after compaction.

---

## Phase 13 — Real-World Workload

### Objective
Test beyond a toy benchmark.

### Experiments
- uneven cluster distributions,
- duplicate vectors,
- highly similar vectors,
- high-dimensional vectors,
- growing indexes,
- different query distributions.

### Exit Criteria
Known failure modes are documented.

---

## Phase 14 — HNSW (Advanced)

### Objective
Implement a second ANN algorithm from scratch.

### Concepts
- graph nodes,
- multiple layers,
- neighbor selection,
- greedy traversal,
- search beam,
- construction parameters.

### Important
HNSW is a separate major project phase, not a prerequisite for MVP.

### Exit Criteria
HNSW can be compared against:
- brute force,
- IVF-Flat.

---

## Phase 15 — Final Comparative Study

### Compare

```text
Exact
  vs
IVF-Flat
  vs
HNSW
```

Across:
- recall,
- latency,
- memory,
- build time,
- insertion cost,
- deletion behavior.

### Final Deliverable

A technical report explaining:

1. what each index does,
2. where approximation enters,
3. why recall changes,
4. why latency changes,
5. what the indexes cost in memory,
6. what workloads favor each design.

---

# Recommended Build Order

```text
Foundation
    ->
Cosine Similarity
    ->
Brute Force
    ->
Common API
    ->
50k Dataset + Ground Truth
    ->
K-Means
    ->
Inverted Lists
    ->
IVF-Flat
    ->
Recall/Latency Benchmarks
    ->
Deletion Semantics
    ->
Performance Optimization
    ->
Persistence
    ->
Compaction
    ->
HNSW
    ->
Final Comparison
```

# MVP Boundary

Stop after Phase 8 if the objective is primarily learning.

At that point you already have the important experiment:

```text
Exact Search
     │
     │ ground truth
     \/
IVF-Flat
     │
     ├── fewer candidates
     ├── lower latency
     └── some recall loss
```

Everything after Phase 8 is an extension, not required to prove the core idea.
