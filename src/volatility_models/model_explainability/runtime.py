"""Runtime objects for the explainability dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.config import PROJECT_ROOT_PATH
from src.config.config import SRC_DIR_PATH
from src.config.config import VOLATILITY_DATA_STEP_DIR_PATH
from src.config.config import config as project_config
from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureDefinition,
    FeatureSchema,
)
from src.volatility_models.model_explainability.services.shared.metrics_registry import (
    MetricDefinition,
    MetricsRegistry,
)


@dataclass(frozen=True)
class ExplainabilityRuntimeSettings:
    """High-level runtime settings resolved from project configuration."""

    project_root: Path
    model_dir: Path
    volatility_dataset_path: Path
    model_input_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numerical_features: tuple[str, ...]
    optional_derived_explainability_features: tuple[str, ...]
    target_column: str
    error_metrics: tuple[str, ...]
    random_state: int
    surrogate_depths: tuple[int, ...]
    surrogate_max_depth: int
    surrogate_min_samples_leaf: int
    surrogate_sample_size: int
    shap_background_size: int
    shap_explain_size: int
    shap_permutations: int
    neighbors_sample_size: int
    diagnosis_sample_size: int
    surface_grid_size: int
    ice_sample_size: int
    curve_points: int
    cache_entries: int


def _r2_score(y_true, y_pred) -> float:
    residual_sum = float(((y_true - y_pred) ** 2).sum())
    centered = y_true - y_true.mean()
    total_sum = float((centered**2).sum())
    if total_sum == 0.0:
        return 0.0
    return 1.0 - residual_sum / total_sum


def build_runtime_settings() -> ExplainabilityRuntimeSettings:
    dashboard_config = project_config.dashboard_models_config
    return ExplainabilityRuntimeSettings(
        project_root=PROJECT_ROOT_PATH,
        model_dir=SRC_DIR_PATH / "volatility_models" / "saved_models",
        volatility_dataset_path=VOLATILITY_DATA_STEP_DIR_PATH
        / f"{project_config.data_config.volatility_config.output_filename}.csv",
        model_input_features=tuple(dashboard_config.model_input_features),
        categorical_features=tuple(dashboard_config.categorical_features),
        numerical_features=tuple(dashboard_config.numerical_features),
        optional_derived_explainability_features=tuple(
            dashboard_config.optional_derived_explainability_features
        ),
        target_column=str(dashboard_config.target_column),
        error_metrics=tuple(dashboard_config.error_metrics),
        random_state=int(dashboard_config.random_state),
        surrogate_depths=tuple(int(depth) for depth in dashboard_config.surrogate_depths),
        surrogate_max_depth=int(dashboard_config.surrogate_max_depth),
        surrogate_min_samples_leaf=int(dashboard_config.surrogate_min_samples_leaf),
        surrogate_sample_size=int(dashboard_config.surrogate_sample_size),
        shap_background_size=int(dashboard_config.shap_background_size),
        shap_explain_size=int(dashboard_config.shap_explain_size),
        shap_permutations=int(dashboard_config.shap_permutations),
        neighbors_sample_size=int(dashboard_config.neighbors_sample_size),
        diagnosis_sample_size=int(dashboard_config.diagnosis_sample_size),
        surface_grid_size=int(dashboard_config.surface_grid_size),
        ice_sample_size=int(dashboard_config.ice_sample_size),
        curve_points=int(dashboard_config.curve_points),
        cache_entries=int(dashboard_config.cache_entries),
    )


def build_feature_schema(settings: ExplainabilityRuntimeSettings) -> FeatureSchema:
    features = [
        FeatureDefinition(
            name="TimeToExpiration",
            label="Time To Expiration (days)",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
            description="Calendar days remaining to maturity.",
        ),
        FeatureDefinition(
            name="Rate",
            label="Rate (%)",
            dtype="float",
            category="numerical",
            raw_input=True,
            widget="number",
            description="Compounded interest rate used in the volatility builder.",
        ),
        FeatureDefinition(
            name="UnderlyingPrice",
            label="Underlying",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
        ),
        FeatureDefinition(
            name="StrikePrice",
            label="Strike",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
        ),
        FeatureDefinition(
            name="OptionType",
            label="Option Type",
            dtype="category",
            category="categorical",
            raw_input=True,
            allowed_values=("C", "P"),
            default_value="C",
            widget="dropdown",
        ),
        FeatureDefinition(
            name="ExecHour",
            label="Execution Hour",
            dtype="int",
            category="numerical",
            raw_input=True,
            min_value=0,
            max_value=23,
            default_value=10,
            widget="slider",
        ),
        FeatureDefinition(
            name="ExecWeekday",
            label="Execution Weekday",
            dtype="int",
            category="categorical",
            raw_input=True,
            allowed_values=(1, 2, 3, 4, 5, 6, 7),
            default_value=1,
            widget="dropdown",
        ),
        FeatureDefinition(
            name="Moneyness",
            label="Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
            min_value=0.0,
            widget="number",
            description="Underlying divided by strike.",
        ),
        FeatureDefinition(
            name="LogMoneyness",
            label="Log-Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
            widget="number",
        ),
        FeatureDefinition(
            name="AbsLogMoneyness",
            label="Absolute Log-Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
            widget="number",
        ),
        FeatureDefinition(
            name="UnderlyingLagMinutes",
            label="Underlying Lag (minutes)",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
            widget="number",
        ),
    ]
    return FeatureSchema(features=features, target_column=settings.target_column)


def build_metrics_registry() -> MetricsRegistry:
    registry = MetricsRegistry()
    registry.register(
        MetricDefinition(
            name="rmse",
            label="RMSE",
            function=lambda y_true, y_pred: float((((y_true - y_pred) ** 2).mean()) ** 0.5),
            higher_is_better=False,
            formatter=lambda value: f"{value:,.4f}",
            description="Root mean squared error.",
        )
    )
    registry.register(
        MetricDefinition(
            name="mae",
            label="MAE",
            function=lambda y_true, y_pred: float((y_true - y_pred).abs().mean()),
            higher_is_better=False,
            formatter=lambda value: f"{value:,.4f}",
            description="Mean absolute error.",
        )
    )
    registry.register(
        MetricDefinition(
            name="r2",
            label="R2",
            function=lambda y_true, y_pred: _r2_score(y_true, y_pred),
            higher_is_better=True,
            formatter=lambda value: f"{value:,.4f}",
            description="Coefficient of determination.",
        )
    )
    return registry


DEFAULT_SETTINGS = build_runtime_settings()
DEFAULT_FEATURE_SCHEMA = build_feature_schema(DEFAULT_SETTINGS)
DEFAULT_METRICS_REGISTRY = build_metrics_registry()
