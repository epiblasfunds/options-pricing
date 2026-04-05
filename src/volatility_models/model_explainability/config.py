"""Central configuration for the model explainability package."""

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

MODEL_INPUT_FEATURES = [
    "TimeToExpiration",
    "Rate",
    "UnderlyingPrice",
    "StrikePrice",
    "OptionType",
    "ExecHour",
    "ExecWeekday",
]

CATEGORICAL_FEATURES = ["OptionType", "ExecWeekday"]
NUMERICAL_FEATURES = [
    "TimeToExpiration",
    "Rate",
    "UnderlyingPrice",
    "StrikePrice",
    "ExecHour",
]
OPTIONAL_DERIVED_EXPLAINABILITY_FEATURES = [
    "Moneyness",
    "LogMoneyness",
    "AbsLogMoneyness",
    "UnderlyingLagMinutes",
]
TARGET_COLUMN = "ImpliedVolatility"
ERROR_METRICS = ["rmse", "mae", "r2"]


@dataclass(frozen=True)
class ExplainabilitySettings:
    """High-level runtime configuration."""

    project_root: Path
    model_dir: Path
    volatility_dataset_path: Path
    model_input_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numerical_features: tuple[str, ...]
    optional_derived_explainability_features: tuple[str, ...]
    target_column: str
    error_metrics: tuple[str, ...]
    random_state: int = 42
    surrogate_max_depth: int = 4
    surrogate_min_samples_leaf: int = 80
    surrogate_sample_size: int = 12000
    shap_background_size: int = 24
    shap_explain_size: int = 48
    shap_permutations: int = 20
    neighbors_sample_size: int = 20000
    diagnosis_sample_size: int = 10000
    surface_grid_size: int = 24
    ice_sample_size: int = 24
    curve_points: int = 25
    cache_entries: int = 64


def _r2_score(y_true, y_pred) -> float:
    residual_sum = float(((y_true - y_pred) ** 2).sum())
    centered = y_true - y_true.mean()
    total_sum = float((centered**2).sum())
    if total_sum == 0.0:
        return 0.0
    return 1.0 - residual_sum / total_sum


def _build_default_feature_schema() -> FeatureSchema:
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
    return FeatureSchema(features=features, target_column=TARGET_COLUMN)


def _build_default_metrics_registry() -> MetricsRegistry:
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
            label="R²",
            function=lambda y_true, y_pred: _r2_score(y_true, y_pred),
            higher_is_better=True,
            formatter=lambda value: f"{value:,.4f}",
            description="Coefficient of determination.",
        )
    )
    return registry


DEFAULT_SETTINGS = ExplainabilitySettings(
    project_root=PROJECT_ROOT_PATH,
    model_dir=SRC_DIR_PATH / "volatility_models" / "saved_models",
    volatility_dataset_path=VOLATILITY_DATA_STEP_DIR_PATH
    / f"{project_config.data_config.volatility_config.output_filename}.csv",
    model_input_features=tuple(MODEL_INPUT_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    numerical_features=tuple(NUMERICAL_FEATURES),
    optional_derived_explainability_features=tuple(
        OPTIONAL_DERIVED_EXPLAINABILITY_FEATURES
    ),
    target_column=TARGET_COLUMN,
    error_metrics=tuple(ERROR_METRICS),
)

DEFAULT_FEATURE_SCHEMA = _build_default_feature_schema()
DEFAULT_METRICS_REGISTRY = _build_default_metrics_registry()
