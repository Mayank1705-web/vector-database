# Vector Database (From First Principles)

A vector database built from scratch for learning and measurement --
no Pinecone, FAISS, Chroma, or sklearn.neighbors. See `.cursorrules`
for the full set of non-negotiable constraints.

Read in this order before changing anything:
1. `PRD.md` -- what we're building and why.
2. `architecture.md` -- component boundaries and data flow.
3. `design.md` -- algorithm-level design decisions.
4. `phases.md` -- the build order and exit criteria for each phase.

## Current status

Phases 0-3 are implemented and tested:
- Phase 0: repository skeleton (this).
- Phase 1: vector validation + cosine similarity (`core/distance.py`, `core/vector.py`).
- Phase 2: exact brute-force index, the correctness oracle (`exact/brute_force.py`).
- Phase 3: common `insert/search/delete` API (`api/index.py`).

Phases 4+ (dataset/ground truth, k-means, IVF-Flat, benchmarking,
persistence, HNSW, ...) are intentionally stubbed with
`NotImplementedError` rather than faked -- see `phases.md` for the
plan and `docs/Handover.md` for exactly where to pick up.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                     # everything
pytest tests/unit          # fast unit tests only
pytest tests/integration   # common-API integration test
pytest tests/benchmarks    # currently smoke-tests evaluation primitives
```

## Quick start

```python
from vector_db.api.index import VectorIndex
from vector_db.exact.brute_force import BruteForceIndex

index = VectorIndex(BruteForceIndex())
index.insert(1, [1.0, 0.0, 0.0])
index.insert(2, [0.0, 1.0, 0.0])

results = index.search([0.9, 0.1, 0.0], k=2)
for r in results:
    print(r.id, r.score)
```

## Repository layout

```text
src/vector_db/
├── core/          # vector/type definitions, distance, validation
├── exact/         # brute-force index (ground truth)
├── approximate/   # IVF-Flat (Phases 5-7, not yet implemented)
├── storage/       # in-memory vector storage
├── api/           # common insert/search/delete abstraction
└── evaluation/    # recall, latency, benchmarking

tests/
├── unit/          # per-module tests
├── integration/   # common-API tests
└── benchmarks/    # performance/evaluation tests

docs/              # decisions, handover notes, flow, constraints
experiments/       # exploratory notebooks + results (gitignored)
data/              # raw/processed data (gitignored)
scripts/           # dataset generation, index building, benchmarking
configs/           # benchmark configuration (default.yaml)
```

See `docs/Handover.md` for a running log of what's done and what's next.