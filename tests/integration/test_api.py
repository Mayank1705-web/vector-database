from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from vector_db.api import server


client = TestClient(server.app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_insert_search_delete() -> None:
    vector = np.zeros(128, dtype=float)
    vector[0] = 1.0

    insert_response = client.post(
        "/insert",
        json={
            "id": 999999,
            "vector": vector.tolist(),
        },
    )

    assert insert_response.status_code == 200
    assert insert_response.json() == {
        "status": "inserted",
        "id": 999999,
    }

    search_response = client.post(
        "/search",
        json={
            "vector": vector.tolist(),
            "k": 1,
        },
    )

    assert search_response.status_code == 200

    results = search_response.json()

    assert len(results) == 1
    assert results[0]["id"] == 999999
    assert results[0]["score"] > 0.99

    delete_response = client.delete("/delete/999999")

    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "status": "deleted",
        "id": 999999,
    }


def test_search_rejects_invalid_k() -> None:
    response = client.post(
        "/search",
        json={
            "vector": [1.0] * 128,
            "k": 0,
        },
    )

    assert response.status_code == 422


def test_insert_rejects_wrong_dimension() -> None:
    response = client.post(
        "/insert",
        json={
            "id": 888888,
            "vector": [1.0] * 127,
        },
    )

    assert response.status_code == 400