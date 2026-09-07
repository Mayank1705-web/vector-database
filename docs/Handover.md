# Handover

Purpose: a running log so any contributor (human or AI) can pick up
the project without re-deriving context. Update this whenever a phase
completes or a session ends mid-phase.

## Status as of repository creation

**Done (Phases 0-3):**
- Repository skeleton matches the structure in `phases.md` / `architecture.md`.
- `core/types.py`, `core/vector.py`, `core/distance.py` -- cosine
  similarity + batch cosine similarity, explicit zero-vector rule
  (score 0.0), dimension-mismatch validation.
- `storage/memory.py` -- in-memory storage with stable ids,
  active/deleted flags, fixed dimensionality enforced on insert.
- `exact/brute_force.py` -- exact top-k cosine search, deterministic
  tie-breaking (score desc, then id asc), the correctness oracle.
- `api/index.py` -- `VectorIndex` wraps any backend satisfying
  `insert/search/delete`; `BruteForceIndex` verified to work entirely
  through this API (see `tests/integration/test_index_api.py`).
- Unit tests for distance and brute force cover: identical/orthogonal/
  opposite/zero vectors, dimension mismatch, k > available vectors,
  empty index, delete, duplicate id, reinsert after delete, negative
  and duplicate vectors.
- `evaluation/recall.py` and `evaluation/latency.py` implemented and
  tested (used later by Phase 8, but they don't depend on IVF).

**Explicitly NOT done (stubbed with `NotImplementedError`, not faked):**
- `scripts/generate_data.py` -- Phase 4 (50k-vector dataset + ground truth).
- `approximate/ivf/clustering.py` -- Phase 5 (k-means).
- `approximate/ivf/inverted_lists.py` -- Phase 6.
- `approximate/ivf/index.py` (`IVFFlatIndex`) -- Phase 7.
- `evaluation/benchmark.py`, `scripts/benchmark.py` -- Phase 8 (recall/latency sweep).
- Everything from Phase 9 onward (API completeness hardening,
  performance engineering, persistence, compaction, real-world
  workloads, HNSW, final comparative study).

## Next logical step

Phase 4: build the dataset + ground truth
(`scripts/generate_data.py`), using `configs/default.yaml` for
`n_vectors` / `n_queries` / `dim` / `seed`. Prefer a real text corpus;
fall back to deterministic clustered synthetic vectors and document
the limitation in `docs/Decisions.md` if you do.

After that: Phase 5 (k-means) -> Phase 6 (inverted lists) -> Phase 7
(IVF-Flat search) -> Phase 8 (measurement). Do not skip ahead per
`.cursorrules` scope discipline -- one phase/logical change at a time.

## How to verify this handover is still accurate

```bash
pytest -q
```

All Phase 0-3 tests should PASS; the clustering/IVF placeholder tests
should PASS by asserting `NotImplementedError` (see `tests/unit/test_clustering.py`,
`tests/unit/test_ivf.py`). If either assumption breaks, update this file.