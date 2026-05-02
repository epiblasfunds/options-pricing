from src.dashboard.services.shared.feature_schema import (
    FeatureDefinition,
    FeatureSchema,
)
from src.dashboard.services.shared.metrics_registry import (
    MetricDefinition,
    MetricsRegistry,
)
from src.enums.data_enums import OptionTypeEnum
from src.model2dashboard.features import RAW_INPUT_FEATURE_NAMES
from src.model2dashboard.features import MODEL_INPUT_FEATURE_NAMES
from src.model2dashboard.features import TARGET_COLUMN
from src.model2dashboard.features import VISIBLE_RAW_INPUT_FEATURE_NAMES


def _r2_score(y_true, y_pred) -> float:
    residual_sum = float(((y_true - y_pred) ** 2).sum())
    centered = y_true - y_true.mean()
    total_sum = float((centered**2).sum())
    if total_sum == 0.0:
        return 0.0
    return 1.0 - residual_sum / total_sum


def build_feature_schema() -> FeatureSchema:
    visible_raw_inputs = set(VISIBLE_RAW_INPUT_FEATURE_NAMES)
    features = [
        _raw_feature_definition(name, raw_input=name in visible_raw_inputs)
        for name in RAW_INPUT_FEATURE_NAMES
    ]
    features.extend(
        [
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
        ]
    )
    known = {feature.name for feature in features}
    for feature_name in MODEL_INPUT_FEATURE_NAMES:
        if feature_name not in known:
            features.append(_model_feature_definition(feature_name))
            known.add(feature_name)
    return FeatureSchema(features=features, target_column=TARGET_COLUMN)


def _raw_feature_definition(
    name: str,
    raw_input: bool = True,
) -> FeatureDefinition:
    labels = {
        "OptionType": "Option Type",
        "StrikePrice": "Strike Price",
        "UnderlyingPrice": "Underlying Price",
        "TimeToExpiration": "Time To Expiration (days)",
        "Rate": "Rate",
    }
    descriptions = {
        "OptionType": "Call or put option.",
        "StrikePrice": "Strike price associated with the traded option.",
        "UnderlyingPrice": "Underlying price paired with the option trade.",
        "TimeToExpiration": "Remaining time to maturity, measured in days.",
        "Rate": "Risk-free rate used to back out implied volatility.",
    }
    if name == "OptionType":
        return FeatureDefinition(
            name=name,
            label=labels[name],
            dtype="category",
            category="categorical",
            raw_input=raw_input,
            derived_explainability_feature=not raw_input,
            allowed_values=(OptionTypeEnum.CALL, OptionTypeEnum.PUT),
            default_value=OptionTypeEnum.CALL,
            widget="dropdown" if raw_input else None,
            description=descriptions[name],
        )
    return FeatureDefinition(
        name=name,
        label=labels.get(name, name),
        dtype="float",
        category="numerical",
        raw_input=raw_input,
        derived_explainability_feature=not raw_input,
        min_value=0.0 if name != "Rate" else None,
        widget="number" if raw_input else None,
        description=descriptions.get(name),
    )


def _model_feature_definition(name: str) -> FeatureDefinition:
    labels = {
        "TTEYears": "TTE (years)",
        "sqrtTTEYears": "Sqrt TTE (years)",
        "logMoneyness": "Log-Moneyness",
        "logMoneynessSq": "Log-Moneyness Squared",
        "logMoneynessXSqrtTTE": "Log-Moneyness x Sqrt TTE",
        "logForwardMoneyness": "Log Forward Moneyness",
        "rate": "Rate",
        "isCall": "Is Call",
        "isPut": "Is Put",
    }
    return FeatureDefinition(
        name=name,
        label=labels.get(name, name),
        dtype="float",
        category="numerical",
        raw_input=False,
        derived_explainability_feature=True,
    )


def build_metrics_registry() -> MetricsRegistry:
    registry = MetricsRegistry()
    registry.register(
        MetricDefinition(
            name="rmse",
            label="RMSE",
            function=lambda y_true, y_pred: float(
                (((y_true - y_pred) ** 2).mean()) ** 0.5
            ),
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
