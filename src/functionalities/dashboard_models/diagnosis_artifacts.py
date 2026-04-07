import pandas as pd

from src.config.config import config
from src.dashboard.domain import build_metrics_registry
from src.enums.data_enums import VolatilityDBEnum
from src.python_models.dashboard.artifacts import DiagnosisArtifact
from src.volatility_models import TARGET_COLUMN

METRICS_REGISTRY = build_metrics_registry()


def build_diagnosis_artifact(
    dataset_frame: pd.DataFrame,
    *,
    financial_warnings: list[str],
    sample_frame,
) -> DiagnosisArtifact:
    sampled = sample_frame(
        dataset_frame.dropna(subset=[str(TARGET_COLUMN)]),
        max_rows=config.dashboard_models_config.diagnosis_sample_size,
        random_state=config.dashboard_models_config.random_state,
    )
    metrics = METRICS_REGISTRY.compute_metrics(
        sampled[str(TARGET_COLUMN)].astype(float).reset_index(drop=True),
        sampled["PredictedVolatility"].astype(float).reset_index(drop=True),
        config.dashboard_models_config.error_metrics,
    )
    plot_frame = sample_frame(
        sampled,
        max_rows=min(2500, len(sampled)),
        random_state=config.dashboard_models_config.random_state + 7,
    )
    error_heatmap = (
        sampled.assign(
            moneyness_bin=pd.cut(sampled["Moneyness"], bins=12),
            maturity_bin=pd.cut(
                sampled[str(VolatilityDBEnum.TIME_TO_EXPIRATION)], bins=12
            ),
        )
        .groupby(["moneyness_bin", "maturity_bin"], observed=False)["AbsoluteError"]
        .mean()
        .reset_index()
    )
    return DiagnosisArtifact(
        metrics={str(name): float(value) for name, value in metrics.items()},
        plot_frame=plot_frame,
        error_heatmap=error_heatmap,
        financial_warnings=list(financial_warnings),
    )
