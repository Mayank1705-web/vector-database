"""Integration test: the exact index used entirely through the common API.

Phase 3 exit criteria (phases.md): "The exact index can be used
entirely through the common API."
"""

from vector_db.api.index import VectorIndex
from vector_db.exact.brute_force import BruteForceIndex


def test_exact_index_through_common_api():
    index = VectorIndex(BruteForceIndex())

    index.insert(1, [1.0, 0.0, 0.0])
    index.insert(2, [0.0, 1.0, 0.0])
    index.insert(3, [0.9, 0.1, 0.0])

    results = index.search([1.0, 0.0, 0.0], k=2)
    ids = [r.id for r in results]
    assert ids == [1, 3]

    index.delete(1)
    results = index.search([1.0, 0.0, 0.0], k=2)
    ids = [r.id for r in results]
    assert 1 not in ids