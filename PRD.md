# Product Requirements Document — From-Scratch Vector Database

## 1. Product Summary

Build a small vector database from first principles to understand and demonstrate how vector retrieval systems work internally.

The system will begin with exact brute-force cosine similarity and then introduce a hand-built IVF-Flat approximate nearest-neighbor index.

The project is primarily an engineering-learning and benchmarking project, not a production replacement for FAISS, Pinecone, Chroma, or similar systems.

## 2. Problem

Modern AI systems frequently depend on vector search, but high-level vector database APIs hide the core mechanics:

- vector storage,
- similarity calculation,
- nearest-neighbor ranking,
- indexing,
- candidate pruning,
- approximate search,
- recall/latency tradeoffs,
- deletion,
- memory and scaling constraints.

The project addresses this knowledge gap by implementing the core mechanics directly.

## 3. Goals

### Primary Goals

- Implement exact cosine nearest-neighbor search.
- Build a reusable index API.
- Support insert, search, and delete.
- Implement IVF-Flat without an ANN library.
- Use brute force as ground truth.
- Quantify the approximation cost.
- Understand how index parameters affect recall and latency.
- Produce reproducible benchmark results.

### Secondary Goals

- Build a clean architecture that can support future indexes.
- Make algorithm behavior easy to inspect.
- Provide experiments that explain rather than merely report performance.
- Leave a foundation for later advanced features.

## 4. Non-Goals

The initial project will NOT attempt to be:

- a production distributed vector database,
- a cloud service,
- a replacement for FAISS,
- a replacement for Pinecone,
- a multi-node database,
- a GPU vector-search engine,
- a full metadata/filtering database,
- a production-grade persistence engine,
- a high-concurrency service.

These may become future experiments only after the core implementation is correct.

## 5. Target User

The primary user is a developer who wants to understand vector databases and approximate nearest-neighbor search deeply enough to reason about their internals.

## 6. MVP

The true MVP is intentionally small.

### MVP Requirements

1. Python package with a clean index interface.
2. NumPy-based vector representation.
3. Exact brute-force index.
4. Cosine similarity.
5. `insert`, `search`, and `delete`.
6. At least 50,000 benchmark vectors.
7. 500 query vectors.
8. Exact top-10 ground truth.
9. A manually implemented IVF-Flat index.
10. IVF search controlled by `n_probe`.
11. Recall@10 measurement.
12. Search latency measurement.
13. Candidate-count measurement.
14. Reproducible benchmark script.

### MVP Success Criteria

The project is successful if it can demonstrate, on a documented dataset:

- exact search correctness,
- IVF search correctness relative to exact search,
- measurable latency reduction when fewer candidates are searched,
- measurable recall loss as approximation becomes more aggressive,
- parameter-dependent tradeoffs between recall and latency.

## 7. Full Version

After the MVP, the project may add:

- Recall@1 and Recall@5.
- p50/p95/p99 latency.
- index build-time benchmarks.
- memory measurements.
- incremental insertion.
- centroid retraining experiments.
- persistence.
- batch search.
- configurable distance metrics.
- vector normalization strategies.
- deleted-vector compaction.
- concurrency experiments.
- larger datasets.
- better clustering initialization.
- alternative indexing strategies.
- HNSW as a separate advanced implementation.
- profiling and low-level optimization.

## 8. Functional Requirements

### FR-1: Vector Insert

The system shall accept:
- a stable vector ID,
- a fixed-dimensional NumPy vector.

It shall reject vectors with incompatible dimensions.

### FR-2: Search

The system shall:
- accept a query vector,
- accept `k`,
- return the highest-scoring active vectors,
- expose similarity scores.

### FR-3: Delete

The system shall remove a vector from future search results.

The implementation may use tombstones initially, provided deleted vectors are excluded from results.

### FR-4: Exact Search

Brute force shall compare the query with every active vector.

### FR-5: IVF Search

IVF shall:
- identify the nearest centroids,
- inspect only selected inverted lists,
- calculate exact vector-to-query similarity for candidates,
- return the best candidates.

### FR-6: Evaluation

The benchmark shall compare IVF results to exact results.

## 9. Data Requirements

Minimum benchmark:

- 50,000 vectors.
- 500 queries.
- top-10 ground truth.

Preferred data source:
- 5,000+ real short texts embedded into vectors.

Fallback:
- deterministic clustered synthetic vectors.

If synthetic data is used, its generation process and seed must be recorded.

## 10. Quality Requirements

### Correctness

Exact search must be treated as the oracle.

### Reproducibility

Experiments must record:
- random seed,
- vector dimension,
- number of vectors,
- number of queries,
- number of clusters,
- `n_probe`,
- k,
- dataset source,
- software environment.

### Performance

Performance claims must include measured data.

## 11. Core Metrics

### Recall@k

For each query:

`recall@k = |approximate_top_k ∩ exact_top_k| / k`

Report the mean over all queries.

### Candidate Fraction

`candidate_fraction = candidates_examined / active_vectors`

This shows how much work IVF avoided.

### Latency

Measure:
- total query latency,
- mean,
- p50,
- p95,
- p99 where practical.

### Build Time

Measure IVF construction separately from query time.

## 12. Architecture Constraints

The exact and approximate indexes should share a common interface where practical.

The algorithm implementation must remain separate from:
- benchmark code,
- data generation,
- experiment reporting,
- CLI/script code.

## 13. Risks

### Risk: Synthetic Data Misleads

Random or poorly generated vectors can produce geometry unlike real embeddings.

Mitigation:
- prefer real embeddings,
- document the dataset,
- run multiple distributions where useful.

### Risk: Python Overhead Dominates

A correct algorithm can appear slow because of Python loops.

Mitigation:
- use NumPy for arithmetic,
- profile before optimizing,
- distinguish algorithmic cost from interpreter overhead.

### Risk: IVF Parameters Hide the Tradeoff

A single configuration can make IVF look arbitrarily good or bad.

Mitigation:
- sweep `n_probe`,
- report recall alongside latency.

### Risk: Deletion Becomes Inconsistent

Deleted vectors may remain in inverted lists.

Mitigation:
- maintain active/deleted state,
- filter deleted candidates,
- later add compaction.

## 14. Product Philosophy

The project should optimize for understanding first and raw performance second.

A slower implementation that exposes the algorithm clearly is more valuable than a highly optimized implementation that becomes impossible to reason about.

## 15. Future Direction

If the project reaches a stable IVF implementation, the next meaningful extension is HNSW.

That extension should be implemented separately so that IVF remains a clear reference point for comparison.

## 16. Definition of Done

The MVP is done when:

- exact index works,
- IVF-Flat works,
- insert/search/delete are tested,
- 50k+ vectors are supported,
- 500 queries are evaluated,
- exact top-10 ground truth exists,
- recall is measured,
- latency is measured,
- candidate reduction is measured,
- benchmark configuration is reproducible,
- actual tests have been run,
- the implementation has been compared against the exact oracle.
