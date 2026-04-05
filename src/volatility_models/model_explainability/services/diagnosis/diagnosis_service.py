"""Model-diagnosis access backed by dashboard artifacts."""

from __future__ import annotations

import pandas as pd

from src.volatility_models.model_explainability.services.behaviour_surface.surface_service import (
    SurfaceService,
)
from src.volatility_models.model_explainability.services.shared.metrics_registry import (
    MetricsRegistry,
)
from src.volatility_models.model_explainability.services.shared.prediction_service import (
    PredictionService,
)


class DiagnosisService:
    """Expose precomputed diagnosis artifacts for the selected model."""

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
        bundle = self.prediction_service.load_bundle(model_id)
        return {
            "metrics": dict(bundle.dashboard_model.diagnosis.metrics),
            "diagnosis_frame": bundle.dashboard_model.dataset_frame.copy(),
            "plot_frame": bundle.dashboard_model.diagnosis.plot_frame.copy(),
            "error_heatmap": bundle.dashboard_model.diagnosis.error_heatmap.copy(),
            "financial_warnings": list(bundle.dashboard_model.diagnosis.financial_warnings),
            "local_surface": pd.DataFrame(),
        }
