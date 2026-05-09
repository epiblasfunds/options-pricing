import pandas as pd

from src.functionalities.dashboard_models.diagnosis_artifacts import (
    build_diagnosis_artifact,
)
from src.model2dashboard.features import TARGET_COLUMN


def test_build_diagnosis_artifact_uses_full_test_for_metrics_and_heatmap():
    dataset = pd.DataFrame(
        {
            TARGET_COLUMN: [0.10, 0.20, 0.40],
            "PredictedVolatility": [0.10, 0.25, 0.30],
            "Moneyness": [0.90, 1.00, 1.10],
            "TimeToExpiration": [10.0, 20.0, 30.0],
            "AbsoluteError": [0.00, 0.05, 0.10],
        }
    )

    artifact = build_diagnosis_artifact(
        dataset,
        financial_warnings=[],
        sample_frame=lambda frame, max_rows, random_state: frame.head(1).copy(),
    )

    assert artifact.metrics["rmse"] > 0.0
    assert len(artifact.plot_frame) == 1
    assert int(artifact.error_heatmap["AbsoluteError"].notna().sum()) == 3
