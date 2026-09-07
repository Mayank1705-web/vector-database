"""In-memory storage for vector records."""

from __future__ import annotations

from vector_db.core.types import VectorRecord


class MemoryStorage:
    """Store active VectorRecord objects in memory."""

    def __init__(self) -> None:
        self._records: dict[int, VectorRecord] = {}

    def insert(self, record: VectorRecord) -> None:
        """Insert a new vector record.

        Raises
        ------
        ValueError
            If the ID is already present.
        """
        if record.id in self._records:
            raise ValueError(f"Vector ID {record.id} already exists.")

        self._records[record.id] = record

    def get(self, vector_id: int) -> VectorRecord:
        """Return the record associated with an ID.

        Raises
        ------
        KeyError
            If the ID does not exist.
        """
        if vector_id not in self._records:
            raise KeyError(f"Vector ID {vector_id} not found.")

        return self._records[vector_id]

    def delete(self, vector_id: int) -> None:
        """Delete a vector record.

        Raises
        ------
        KeyError
            If the ID does not exist.
        """
        if vector_id not in self._records:
            raise KeyError(f"Vector ID {vector_id} not found.")

        del self._records[vector_id]

    def exists(self, vector_id: int) -> bool:
        """Return whether an active record exists for the ID."""
        return vector_id in self._records

    def records(self) -> list[VectorRecord]:
        """Return all active vector records."""
        return list(self._records.values())

    @property
    def count(self) -> int:
        """Return the number of active vector records."""
        return len(self._records)


__all__ = ["MemoryStorage"]