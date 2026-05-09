import pandas as pd

from src.config.config import config
from src.dashboard.domain import build_metrics_registry
from src.dashboard.utils.diagnosis import build_error_heatmap_frame
from src.enums.data_enums import VolatilityDBEnum
from src.model2dashboard.features import TARGET_COLUMN
from src.python_models.dashboard.artifacts import DiagnosisArtifact

METRICS_REGISTRY = build_metrics_registry()


def build_diagnosis_artifact(
    dataset_frame: pd.DataFrame,
    *,
    financial_warnings: list[str],
    sample_frame,
) -> DiagnosisArtifact:
    evaluation_frame = dataset_frame.dropna(subset=[str(TARGET_COLUMN)]).copy()
    metrics = METRICS_REGISTRY.compute_metrics(
        evaluation_frame[str(TARGET_COLUMN)].astype(float).reset_index(drop=True),
        evaluation_frame["PredictedVolatility"].astype(float).reset_index(drop=True),
        config.dashboard_models_config.error_metrics,
    )
    plot_frame = sample_frame(
        evaluation_frame,
        max_rows=min(2500, len(evaluation_frame)),
        random_state=config.dashboard_models_config.random_state + 7,
    )
    error_heatmap = build_error_heatmap_frame(evaluation_frame)
    return DiagnosisArtifact(
        metrics={str(name): float(value) for name, value in metrics.items()},
        plot_frame=plot_frame,
        error_heatmap=error_heatmap,
        financial_warnings=list(financial_warnings),
    )
