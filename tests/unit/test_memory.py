"""Unit tests for MemoryStorage."""

import numpy as np
import pytest

from vector_db.core.types import VectorRecord
from vector_db.storage.memory import MemoryStorage


def make_record(vector_id: int) -> VectorRecord:
    """Create a test VectorRecord."""
    return VectorRecord(
        id=vector_id,
        vector=np.array([1.0, 2.0, 3.0]),
    )


def test_insert() -> None:
    storage = MemoryStorage()
    record = make_record(1)

    storage.insert(record)

    assert storage.get(1) == record


def test_duplicate_ids_raise_value_error() -> None:
    storage = MemoryStorage()

    storage.insert(make_record(1))

    with pytest.raises(ValueError):
        storage.insert(make_record(1))


def test_get() -> None:
    storage = MemoryStorage()
    record = make_record(42)

    storage.insert(record)

    result = storage.get(42)

    assert result == record
    assert result.id == 42


def test_get_missing_id_raises_key_error() -> None:
    storage = MemoryStorage()

    with pytest.raises(KeyError):
        storage.get(999)


def test_delete() -> None:
    storage = MemoryStorage()
    storage.insert(make_record(1))

    storage.delete(1)

    assert not storage.exists(1)
    assert storage.count == 0


def test_delete_missing_id_raises_key_error() -> None:
    storage = MemoryStorage()

    with pytest.raises(KeyError):
        storage.delete(999)


def test_exists() -> None:
    storage = MemoryStorage()

    assert not storage.exists(1)

    storage.insert(make_record(1))

    assert storage.exists(1)

    storage.delete(1)

    assert not storage.exists(1)


def test_records() -> None:
    storage = MemoryStorage()

    record_1 = make_record(1)
    record_2 = make_record(2)

    storage.insert(record_1)
    storage.insert(record_2)

    records = storage.records()

    assert len(records) == 2
    assert record_1 in records
    assert record_2 in records


def test_count() -> None:
    storage = MemoryStorage()

    assert storage.count == 0

    storage.insert(make_record(1))
    assert storage.count == 1

    storage.insert(make_record(2))
    assert storage.count == 2

    storage.delete(1)
    assert storage.count == 1


def test_reinsert_deleted_id() -> None:
    storage = MemoryStorage()

    first_record = make_record(1)
    second_record = VectorRecord(
        id=1,
        vector=np.array([4.0, 5.0, 6.0]),
    )

    storage.insert(first_record)
    storage.delete(1)

    storage.insert(second_record)

    assert storage.exists(1)
    assert storage.get(1) == second_record
    assert storage.count == 1

def test_records_returns_separate_list() -> None:
    storage = MemoryStorage()

    record_1 = make_record(1)
    record_2 = make_record(2)

    storage.insert(record_1)
    storage.insert(record_2)

    records = storage.records()
    records_again = storage.records()

    assert records == records_again
    assert records is not records_again


def test_mutating_records_list_does_not_change_storage() -> None:
    storage = MemoryStorage()

    record_1 = make_record(1)
    record_2 = make_record(2)

    storage.insert(record_1)
    storage.insert(record_2)

    records = storage.records()

    records.clear()

    assert storage.count == 2
    assert storage.exists(1)
    assert storage.exists(2)
    assert storage.get(1) == record_1
    assert storage.get(2) == record_2

def test_appending_to_records_list_does_not_change_storage() -> None:
    storage = MemoryStorage()

    record_1 = make_record(1)
    record_2 = make_record(2)
    extra_record = make_record(3)

    storage.insert(record_1)
    storage.insert(record_2)

    records = storage.records()
    records.append(extra_record)

    assert storage.count == 2
    assert storage.exists(1)
    assert storage.exists(2)
    assert not storage.exists(3)
    assert storage.get(1) == record_1
    assert storage.get(2) == record_2