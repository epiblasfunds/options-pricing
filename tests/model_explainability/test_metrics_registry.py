import pandas as pd

from src.volatility_models.model_explainability.config import DEFAULT_METRICS_REGISTRY


def test_metrics_registry_computes_configured_metrics():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 1.5, 2.5])

    metrics = DEFAULT_METRICS_REGISTRY.compute_metrics(y_true, y_pred, ["rmse", "mae", "r2"])

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert DEFAULT_METRICS_REGISTRY.format_metric("rmse", metrics["rmse"])
