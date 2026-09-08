"""Tests for the semantic search demo."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.semantic_search_demo import (
    MODEL_NAME,
    SAMPLE_STATEMENTS,
    build_demo_index,
    load_embedding_model,
    main,
    text_to_vector,
)


@pytest.fixture(scope="module")
def embedding_model():
    """Load the real embedding model once for the test module."""
    return load_embedding_model()


def test_embedding_model_loads() -> None:
    """The configured sentence embedding model should load successfully."""
    model = load_embedding_model()

    assert model is not None


def test_text_to_vector_is_deterministic(embedding_model) -> None:
    """The same text should produce the same embedding."""
    text = "I enjoy programming and building software."

    first = text_to_vector(text, embedding_model)
    second = text_to_vector(text, embedding_model)

    np.testing.assert_allclose(first, second)


def test_text_to_vector_has_expected_shape(embedding_model) -> None:
    """The embedding should match the model's output dimension."""
    vector = text_to_vector("I enjoy programming.", embedding_model)

    expected_dimension = embedding_model.get_embedding_dimension()

    assert vector.shape == (expected_dimension,)
    assert vector.dtype == np.float64


def test_text_to_vector_is_normalized(embedding_model) -> None:
    """Generated embeddings should have unit length."""
    vector = text_to_vector("I enjoy programming.", embedding_model)

    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_text_to_vector_is_nonzero(embedding_model) -> None:
    """Valid text should produce a non-zero embedding."""
    vector = text_to_vector("vector database", embedding_model)

    assert np.linalg.norm(vector) > 0.0


def test_text_to_vector_rejects_non_string(embedding_model) -> None:
    """Non-string input should be rejected."""
    with pytest.raises(TypeError, match="text must be a string"):
        text_to_vector(123, embedding_model)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   "])
def test_text_to_vector_rejects_empty_text(
    embedding_model,
    text: str,
) -> None:
    """Empty text should be rejected."""
    with pytest.raises(ValueError, match="text must not be empty"):
        text_to_vector(text, embedding_model)


def test_build_demo_index_contains_all_statements(
    embedding_model,
) -> None:
    """The demo index should contain every sample statement."""
    index, statements, dimension = build_demo_index(embedding_model)

    assert len(statements) == len(SAMPLE_STATEMENTS)
    assert list(statements.values()) == SAMPLE_STATEMENTS
    assert dimension == embedding_model.get_embedding_dimension()

    results = index.search(
        text_to_vector(SAMPLE_STATEMENTS[0], embedding_model),
        k=1,
    )

    assert len(results) == 1
    assert results[0].id == 0


def test_build_demo_index_returns_searchable_ivf_index(
    embedding_model,
) -> None:
    """The demo index should perform IVF-Flat search."""
    index, statements, _ = build_demo_index(embedding_model)

    query = "I like programming and writing code."

    results = index.search(
        text_to_vector(query, embedding_model),
        k=3,
    )

    assert len(results) == 3
    assert all(result.id in statements for result in results)
    assert all(np.isfinite(result.score) for result in results)

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_build_demo_index_is_deterministic(
    embedding_model,
) -> None:
    """Repeated index construction should produce the same search results."""
    first_index, _, _ = build_demo_index(embedding_model)
    second_index, _, _ = build_demo_index(embedding_model)

    query = text_to_vector(
        "I enjoy developing software.",
        embedding_model,
    )

    first_results = first_index.search(query, k=3)
    second_results = second_index.search(query, k=3)

    assert first_results == second_results


def test_model_name_is_configured() -> None:
    """The demo should expose the configured embedding model name."""
    assert MODEL_NAME == "all-MiniLM-L6-v2"


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
    assert "Embedding model:" in output
    assert "Vector database: IVF-Flat" in output
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
            "",
            "exit",
        ]
    )

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out

    assert "Please enter a non-empty query." in output
    assert "Demo finished." in output


def test_main_handles_model_loading_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should report an embedding-model loading failure."""

    def fail_to_load_model():
        raise RuntimeError("model loading failed")

    monkeypatch.setattr(
        "scripts.semantic_search_demo.load_embedding_model",
        fail_to_load_model,
    )

    main()

    output = capsys.readouterr().out

    assert "Loading embedding model..." in output
    assert "Failed to load embedding model: model loading failed" in output