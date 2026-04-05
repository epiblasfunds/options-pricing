"""Model-diagnosis computations."""

from __future__ import annotations

import pandas as pd

from src.volatility_models.model_explainability.config import DEFAULT_SETTINGS
from src.volatility_models.model_explainability.services.behaviour_surface.surface_service import (
    SurfaceService,
)
from src.volatility_models.model_explainability.services.shared.metrics_registry import (
    MetricsRegistry,
)
from src.volatility_models.model_explainability.services.shared.prediction_service import (
    PredictionService,
)
from src.volatility_models.model_explainability.utils.sampling import sample_frame


class DiagnosisService:
    """Residual analysis and lightweight financial sanity checks."""

    def __init__(
        self,
        prediction_service: PredictionService,
        metrics_registry: MetricsRegistry,
        surface_service: SurfaceService,
        target_column: str,
    ) -> None:
        self.prediction_service = prediction_service
        self.metrics_registry = metrics_registry
        self.surface_service = surface_service
        self.target_column = target_column

    def diagnose(self, model_id: str, frame: pd.DataFrame) -> dict[str, object]:
        sampled = sample_frame(
            frame.dropna(subset=[self.target_column]),
            max_rows=DEFAULT_SETTINGS.diagnosis_sample_size,
            random_state=DEFAULT_SETTINGS.random_state,
        )
        predictions = self.prediction_service.predict_frame(model_id, sampled)
        actual = sampled[self.target_column].astype(float)
        metrics = self.metrics_registry.compute_metrics(
            actual.reset_index(drop=True),
            predictions.reset_index(drop=True),
            DEFAULT_SETTINGS.error_metrics,
        )
        residuals = actual - predictions
        diagnosis_frame = sampled.copy()
        diagnosis_frame["PredictedVolatility"] = predictions
        diagnosis_frame["Residual"] = residuals
        diagnosis_frame["AbsoluteError"] = residuals.abs()
        plot_frame = sample_frame(
            diagnosis_frame,
            max_rows=min(2500, DEFAULT_SETTINGS.diagnosis_sample_size),
            random_state=DEFAULT_SETTINGS.random_state + 7,
        )
        error_heatmap = (
            diagnosis_frame.assign(
                moneyness_bin=pd.cut(diagnosis_frame["Moneyness"], bins=12),
                maturity_bin=pd.cut(diagnosis_frame["TimeToExpiration"], bins=12),
            )
            .groupby(["moneyness_bin", "maturity_bin"], observed=False)["AbsoluteError"]
            .mean()
            .reset_index()
        )
        anchor = self.surface_service.default_anchor(sampled)
        surface_frame = self.surface_service.build_surface(model_id, anchor)
        warnings = self.surface_service.financial_checks(surface_frame)
        return {
            "metrics": metrics,
            "diagnosis_frame": diagnosis_frame,
            "plot_frame": plot_frame,
            "error_heatmap": error_heatmap,
            "financial_warnings": warnings,
            "local_surface": surface_frame,
        }
