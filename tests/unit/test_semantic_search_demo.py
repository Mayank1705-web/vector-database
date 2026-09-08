"""Tests for the semantic search demo."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.semantic_search_demo import (
    DIMENSION,
    SAMPLE_STATEMENTS,
    build_demo_index,
    main,
    text_to_vector,
)


def test_text_to_vector_is_deterministic() -> None:
    """The same text should always produce the same vector."""
    text = "I enjoy programming and building software."

    first = text_to_vector(text)
    second = text_to_vector(text)

    np.testing.assert_array_equal(first, second)


def test_text_to_vector_has_expected_shape() -> None:
    """Generated embeddings should have the configured dimension."""
    vector = text_to_vector("I enjoy programming.")

    assert vector.shape == (DIMENSION,)
    assert vector.dtype == np.float64


def test_text_to_vector_is_normalized() -> None:
    """Generated embeddings should have unit length."""
    vector = text_to_vector("I enjoy programming.")

    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_text_to_vector_is_nonzero() -> None:
    """A valid text should produce a non-zero vector."""
    vector = text_to_vector("vector database")

    assert np.linalg.norm(vector) > 0.0


def test_text_to_vector_rejects_non_string() -> None:
    """Non-string input should be rejected."""
    with pytest.raises(TypeError, match="text must be a string"):
        text_to_vector(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   "])
def test_text_to_vector_rejects_empty_text(text: str) -> None:
    """Empty text should be rejected."""
    with pytest.raises(ValueError, match="text must not be empty"):
        text_to_vector(text)


def test_text_to_vector_rejects_text_without_tokens() -> None:
    """Text without alphanumeric tokens should be rejected."""
    with pytest.raises(
        ValueError,
        match="at least one alphanumeric token",
    ):
        text_to_vector("!!! ???")


def test_build_demo_index_contains_all_statements() -> None:
    """The demo index should contain every sample statement."""
    index, statements = build_demo_index()

    assert len(statements) == len(SAMPLE_STATEMENTS)
    assert list(statements.values()) == SAMPLE_STATEMENTS

    results = index.search(
        text_to_vector(SAMPLE_STATEMENTS[0]),
        k=1,
    )

    assert len(results) == 1
    assert results[0].id == 0


def test_build_demo_index_returns_searchable_ivf_index() -> None:
    """The demo index should perform IVF-Flat search."""
    index, statements = build_demo_index()

    query = "I like programming and writing code."
    results = index.search(text_to_vector(query), k=3)

    assert len(results) == 3
    assert all(result.id in statements for result in results)
    assert all(np.isfinite(result.score) for result in results)

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_build_demo_index_is_deterministic() -> None:
    """Building the demo index repeatedly should produce the same results."""
    first_index, _ = build_demo_index()
    second_index, _ = build_demo_index()

    query = text_to_vector("I enjoy developing software.")

    first_results = first_index.search(query, k=3)
    second_results = second_index.search(query, k=3)

    assert first_results == second_results


def test_main_query_flow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should process a query and exit cleanly."""
    inputs = iter(
        [
            "I enjoy programming and building software",
            "exit",
        ]
    )

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out

    assert "Semantic Search Demo" in output
    assert "Query" in output
    assert "Best matches" in output
    assert "Similarity:" in output
    assert "Demo finished." in output


def test_main_query_flow_handles_empty_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should reject an empty query and continue."""
    inputs = iter(
        [
            "",
            "exit",
        ]
    )

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out

    assert "Please enter a non-empty query." in output
    assert "Demo finished." in output


def test_main_query_flow_handles_invalid_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should report an invalid query without crashing."""
    inputs = iter(
        [
            "!!! ???",
            "exit",
        ]
    )

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out

    assert "Error:" in output
    assert "Demo finished." in output