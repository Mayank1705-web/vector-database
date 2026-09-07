import numpy as np
import pytest

from vector_db.core.distance import batch_cosine_similarity, cosine_similarity


def test_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_normalized_vectors_match_dot_product():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    # dot product of orthonormal vectors is 0, same as cosine similarity
    assert cosine_similarity(a, b) == pytest.approx(np.dot(a, b))


def test_zero_vector_is_defined_as_zero_similarity():
    zero = np.array([0.0, 0.0, 0.0])
    other = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(zero, other) == 0.0
    assert cosine_similarity(zero, zero) == 0.0


def test_dimension_mismatch_raises():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_batch_matches_pairwise():
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    scores = batch_cosine_similarity(query, matrix)
    expected = [cosine_similarity(query, matrix[i]) for i in range(matrix.shape[0])]
    np.testing.assert_allclose(scores, expected)


def test_batch_empty_matrix():
    query = np.array([1.0, 0.0])
    matrix = np.empty((0, 2))
    scores = batch_cosine_similarity(query, matrix)
    assert scores.shape == (0,)


def test_batch_dimension_mismatch_raises():
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        batch_cosine_similarity(query, matrix)