from __future__ import annotations

from src.config.config import config
from src.dashboard.services.shared.feature_schema import FeatureDefinition
from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.dashboard.services.shared.metrics_registry import MetricDefinition
from src.dashboard.services.shared.metrics_registry import MetricsRegistry
from src.enums.data_enums import OptionTypeEnum
from src.enums.data_enums import VolatilityDBEnum
from src.volatility_models.feature_engineering import MODEL_FEATURE_NAMES
from src.volatility_models.feature_engineering import RAW_INPUT_FEATURES
from src.volatility_models.feature_engineering import TRADE_TYPE_TO_FEATURE


def _r2_score(y_true, y_pred) -> float:
    residual_sum = float(((y_true - y_pred) ** 2).sum())
    centered = y_true - y_true.mean()
    total_sum = float((centered**2).sum())
    if total_sum == 0.0:
        return 0.0
    return 1.0 - residual_sum / total_sum


def build_feature_schema() -> FeatureSchema:
    features = [
        FeatureDefinition(
            name=str(VolatilityDBEnum.EXEC_DATETIME),
            label="Execution Datetime",
            dtype="datetime",
            category="categorical",
            raw_input=True,
            widget="text",
            description="Execution timestamp of the option trade.",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.OPTION_TYPE),
            label="Option Type",
            dtype="category",
            category="categorical",
            raw_input=True,
            allowed_values=(OptionTypeEnum.CALL, OptionTypeEnum.PUT),
            default_value=OptionTypeEnum.CALL,
            widget="dropdown",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.QUANTITY),
            label="Quantity",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.STRIKE_PRICE),
            label="Strike Price",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.TRADE_TYPE),
            label="Trade Type",
            dtype="category",
            category="categorical",
            raw_input=True,
            allowed_values=tuple(TRADE_TYPE_TO_FEATURE.keys()),
            default_value="M",
            widget="dropdown",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.UNDERLYING_LAG_MINUTES),
            label="Underlying Lag (minutes)",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.UNDERLYING_PRICE),
            label="Underlying Price",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.TIME_TO_EXPIRATION),
            label="Time To Expiration (days)",
            dtype="float",
            category="numerical",
            raw_input=True,
            min_value=0.0,
            widget="number",
        ),
        FeatureDefinition(
            name=str(VolatilityDBEnum.RATE),
            label="Rate",
            dtype="float",
            category="numerical",
            raw_input=True,
            widget="number",
        ),
        FeatureDefinition(
            name="ExecHour",
            label="Execution Hour",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="ExecWeekday",
            label="Execution Weekday",
            dtype="int",
            category="categorical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="Moneyness",
            label="Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="LogMoneyness",
            label="Log-Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="AbsLogMoneyness",
            label="Absolute Log-Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="tte_years",
            label="TTE (years)",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="sqrt_tte_years",
            label="Sqrt TTE (years)",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="log_moneyness",
            label="Log-Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="log_moneyness_sq",
            label="Log-Moneyness Squared",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="log_moneyness_x_sqrt_tte",
            label="Log-Moneyness x Sqrt TTE",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="log_forward_moneyness",
            label="Log Forward Moneyness",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="rate",
            label="Rate",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="is_call",
            label="Is Call",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="exec_hour",
            label="Execution Hour",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="exec_weekday",
            label="Execution Weekday",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="underlying_lag_minutes",
            label="Underlying Lag (minutes)",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
        FeatureDefinition(
            name="quantity_log1p",
            label="Log(1 + Quantity)",
            dtype="float",
            category="numerical",
            raw_input=False,
            derived_explainability_feature=True,
        ),
    ]
    for trade_type, feature_name in TRADE_TYPE_TO_FEATURE.items():
        features.append(
            FeatureDefinition(
                name=feature_name,
                label=f"Trade Type = {trade_type}",
                dtype="float",
                category="numerical",
                raw_input=False,
                derived_explainability_feature=True,
            )
        )

    return FeatureSchema(
        features=features,
        target_column=str(VolatilityDBEnum.IMPLIED_VOLATILITY),
    )


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


RAW_INPUT_FEATURE_NAMES = [str(feature) for feature in RAW_INPUT_FEATURES]
MODEL_INPUT_FEATURE_NAMES = list(MODEL_FEATURE_NAMES)
ERROR_METRICS = tuple(config.dashboard_models_config.error_metrics)
