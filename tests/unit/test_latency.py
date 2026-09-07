import pytest

from vector_db.evaluation.latency import (
    LatencyStats,
    calculate_percentiles,
)


def test_calculate_percentiles() -> None:
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]

    stats = calculate_percentiles(latencies)

    assert stats.p50 == pytest.approx(3.0)
    assert stats.p95 == pytest.approx(4.8)
    assert stats.p99 == pytest.approx(4.96)


def test_calculate_percentiles_accepts_unsorted_values() -> None:
    latencies = [5.0, 1.0, 4.0, 2.0, 3.0]

    stats = calculate_percentiles(latencies)

    assert stats.p50 == pytest.approx(3.0)


def test_calculate_percentiles_returns_min_max() -> None:
    latencies = [2.0, 5.0, 1.0, 4.0, 3.0]

    stats = calculate_percentiles(latencies)

    assert stats.minimum == 1.0
    assert stats.maximum == 5.0


def test_calculate_percentiles_returns_mean() -> None:
    latencies = [1.0, 2.0, 3.0]

    stats = calculate_percentiles(latencies)

    assert stats.mean == pytest.approx(2.0)


def test_latency_stats_is_immutable() -> None:
    stats = LatencyStats(
        p50=1.0,
        p95=2.0,
        p99=3.0,
        minimum=0.5,
        maximum=4.0,
        mean=1.5,
    )

    with pytest.raises(AttributeError):
        stats.p50 = 10.0


def test_empty_latencies_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        calculate_percentiles([])


def test_negative_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        calculate_percentiles([1.0, -0.1, 2.0])


def test_non_numeric_latency_is_rejected() -> None:
    with pytest.raises(TypeError, match="numeric"):
        calculate_percentiles([1.0, "2.0", 3.0])


def test_nan_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_percentiles([1.0, float("nan"), 3.0])


def test_infinite_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_percentiles([1.0, float("inf"), 3.0])