import pytest

from vector_db.core.types import SearchResult
from vector_db.evaluation.recall import recall_at_k, mean_recall_at_k


def test_recall_at_k_perfect_match() -> None:
    ground_truth = [1, 2, 3, 4, 5]
    approximate = [1, 2, 3, 4, 5]

    assert recall_at_k(ground_truth, approximate, 5) == 1.0


def test_recall_at_k_no_match() -> None:
    ground_truth = [1, 2, 3, 4, 5]
    approximate = [6, 7, 8, 9, 10]

    assert recall_at_k(ground_truth, approximate, 5) == 0.0


def test_recall_at_k_partial_match() -> None:
    ground_truth = [1, 2, 3, 4, 5]
    approximate = [1, 2, 8, 9, 10]

    assert recall_at_k(ground_truth, approximate, 5) == pytest.approx(0.4)


def test_recall_at_k_only_compares_top_k() -> None:
    ground_truth = [1, 2, 3, 4, 5]
    approximate = [1, 2, 8, 9, 3]

    assert recall_at_k(ground_truth, approximate, 2) == 1.0


def test_recall_at_k_supports_search_results() -> None:
    ground_truth = [
        SearchResult(id=1, score=1.0),
        SearchResult(id=2, score=0.9),
        SearchResult(id=3, score=0.8),
    ]

    approximate = [
        SearchResult(id=1, score=1.0),
        SearchResult(id=3, score=0.8),
        SearchResult(id=9, score=0.5),
    ]

    assert recall_at_k(ground_truth, approximate, 3) == pytest.approx(2 / 3)


def test_recall_at_k_duplicate_ids_do_not_inflate_recall() -> None:
    ground_truth = [1, 2, 3]
    approximate = [1, 1, 1]

    assert recall_at_k(ground_truth, approximate, 3) == pytest.approx(1 / 3)


def test_recall_at_k_empty_results() -> None:
    assert recall_at_k([], [], 5) == 0.0


def test_recall_at_k_invalid_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        recall_at_k([1, 2], [1, 2], 0)


def test_mean_recall_at_k() -> None:
    ground_truth = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    approximate = [
        [1, 2, 9],
        [4, 9, 10],
    ]

    assert mean_recall_at_k(ground_truth, approximate, 3) == pytest.approx(0.5)


def test_mean_recall_at_k_multiple_k_values() -> None:
    ground_truth = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]

    approximate = [
        [1, 2, 3, 11, 12],
        [6, 7, 13, 14, 15],
    ]

    assert mean_recall_at_k(ground_truth, approximate, 1) == 1.0
    assert mean_recall_at_k(ground_truth, approximate, 5) == pytest.approx(0.5)