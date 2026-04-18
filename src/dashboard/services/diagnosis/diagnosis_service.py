"""Model-diagnosis access backed by dashboard artifacts."""

import pandas as pd

from src.dashboard.services.shared.prediction_service import PredictionService


class DiagnosisService:
    """Expose precomputed diagnosis artifacts for the selected model."""

    def __init__(
        self,
        prediction_service: PredictionService,
    ) -> None:
        self.prediction_service = prediction_service

    def diagnose(self, model_id: str) -> dict[str, object]:
        bundle = self.prediction_service.load_bundle(model_id)
        return {
            "metrics": dict(bundle.dashboard_model.diagnosis.metrics),
            "diagnosis_frame": bundle.dashboard_model.dataset_frame.copy(),
            "plot_frame": bundle.dashboard_model.diagnosis.plot_frame.copy(),
            "error_heatmap": bundle.dashboard_model.diagnosis.error_heatmap.copy(),
            "financial_warnings": list(
                bundle.dashboard_model.diagnosis.financial_warnings
            ),
            "local_surface": pd.DataFrame(),
        }
