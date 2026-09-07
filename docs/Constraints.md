# Constraints

Restating `.cursorrules` in one place so they're easy to check
against during review. This file should never drift from
`.cursorrules` -- if they disagree, `.cursorrules` wins; fix this file.

## Hard "do not" list

- Do NOT use Pinecone, FAISS, Chroma, or `sklearn.neighbors`.
- Do NOT substitute any other ANN/vector-index library for our own implementation.
- Do NOT let IVF silently fall back to brute force during normal search.
- Do NOT commit large generated datasets (`data/raw/`, `data/processed/`
  are gitignored; commit generation config instead).
- Do NOT combine unrelated work in one change (algorithm rewrite +
  persistence + API redesign + benchmarking + cleanup + dependency
  bump, all at once).
- Do NOT claim a test passed without actually running it.
- Do NOT improve latency by silently reducing correctness.
- Do NOT use vague verification language -- only PASS / FAIL / NOT RUN / NOT VERIFIED.

## Required for every new index

- Compare against brute force.
- Test small, hand-checkable datasets.
- Test duplicate/similar vectors.
- Test negative values.
- Test zero-vector behavior.
- Test k larger than available active vectors.
- Test deletion.
- Test empty indexes.
- Test invalid dimensions.

## Required for every approximate-search result

Evaluate against exact top-k, measuring at minimum: Recall@1, Recall@5,
Recall@10, per-query latency, average latency, p50/p95/p99 latency
(when practical), candidate count / fraction scanned, index build
time, memory usage (when practical). Always state the benchmark
configuration.

## Dependency rule

NumPy is allowed for vector arithmetic/array ops. Any *new* dependency
needs: why NumPy is insufficient, what capability it adds, why the
learning objective requires it, and its maintenance cost. Never add an
ANN library merely to make the benchmark faster.

## MVP data target (Phase 4+)

At least 50,000 vectors, documented dimensionality, 500 query vectors,
exact top-10 ground truth. Prefer real text embeddings; fall back to
deterministic synthetic clustered vectors and document the limitation
if used.