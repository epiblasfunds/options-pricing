import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.dashboard.domain import build_metrics_registry


def test_metrics_registry_computes_and_formats_configured_metrics():
    registry = build_metrics_registry()
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 1.5, 2.5])

    metrics = registry.compute_metrics(y_true, y_pred, ["rmse", "mae", "r2"])

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] == pytest.approx(0.408248290463863)
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert registry.format_metric("rmse", metrics["rmse"]) == "0.4082"
    assert registry.get("r2").higher_is_better is True


def test_metrics_registry_returns_zero_r2_for_constant_targets():
    registry = build_metrics_registry()

    metrics = registry.compute_metrics(
        pd.Series([2.0, 2.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        ["r2"],
    )

    assert metrics["r2"] == 0.0


def test_metrics_registry_ignores_nonfinite_pairs_consistently():
    registry = build_metrics_registry()
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.0, float("nan"), float("inf"), 3.5])

    metrics = registry.compute_metrics(y_true, y_pred, ["rmse", "mae", "r2"])

    expected_true = pd.Series([1.0, 4.0])
    expected_pred = pd.Series([1.0, 3.5])
    assert metrics["rmse"] == pytest.approx((((0.0**2) + (0.5**2)) / 2.0) ** 0.5)
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["r2"] == pytest.approx(
        r2_score(expected_true, expected_pred, force_finite=True)
    )


def test_metrics_registry_returns_nan_when_no_finite_pairs_exist():
    registry = build_metrics_registry()

    metrics = registry.compute_metrics(
        pd.Series([1.0, 2.0]),
        pd.Series([float("nan"), float("inf")]),
        ["rmse", "mae", "r2"],
    )

    assert all(pd.isna(value) for value in metrics.values())
