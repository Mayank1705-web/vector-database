"""K-means-style clustering for IVF-Flat.

NOT YET IMPLEMENTED -- this is Phase 5 (see phases.md). Left as an
explicit stub rather than a silent pass-through, per .cursorrules
("If an operation is intentionally unsupported, document it rather
than silently pretending it works.").

Planned inputs (architecture.md section 8):
    - vectors
    - number of clusters (n_clusters)
    - number of iterations
    - random seed

Planned outputs:
    - centroid matrix, shape (n_clusters, dim)
    - vector-to-cluster assignment array, shape (n_vectors,)
"""

from __future__ import annotations


def fit_kmeans(vectors, n_clusters: int, n_iter: int = 10, seed: int = 0):
    """Placeholder. Implement in Phase 5."""
    raise NotImplementedError(
        "K-means clustering is Phase 5 work (see phases.md) and has not "
        "been implemented yet."
    )


__all__ = ["fit_kmeans"]