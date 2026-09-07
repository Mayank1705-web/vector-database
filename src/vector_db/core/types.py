from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


VectorArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SearchResult:
    """One vector-search result."""

    id: int
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, int):
            raise TypeError("SearchResult.id must be an int.")

        if not isinstance(self.score, (int, float, np.integer, np.floating)):
            raise TypeError("SearchResult.score must be numeric.")

        score = float(self.score)

        if not np.isfinite(score):
            raise ValueError("SearchResult.score must be finite.")

        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class VectorRecord:
    """One stored vector and its stable identifier."""

    id: int
    vector: VectorArray

    def __post_init__(self) -> None:
        if not isinstance(self.id, int):
            raise TypeError("VectorRecord.id must be an int.")

        vector = np.asarray(self.vector, dtype=np.float64)

        if vector.ndim != 1:
            raise ValueError("VectorRecord.vector must be a 1-D vector.")

        if vector.size == 0:
            raise ValueError("VectorRecord.vector must not be empty.")

        if not np.all(np.isfinite(vector)):
            raise ValueError("VectorRecord.vector must contain only finite values.")

        vector = np.array(vector, dtype=np.float64, copy=True)
        vector.setflags(write=False)

        object.__setattr__(self, "vector", vector)


__all__ = [
    "SearchResult",
    "VectorArray",
    "VectorRecord",
]