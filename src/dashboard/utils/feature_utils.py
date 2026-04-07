"""Feature helpers shared by the dashboard and bundle exporter."""

import pandas as pd

from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.volatility_models import add_dashboard_derived_features, apply_feature_override


def add_derived_features(
    frame: pd.DataFrame, feature_schema: FeatureSchema
) -> pd.DataFrame:
    del feature_schema
    return add_dashboard_derived_features(frame)


def build_sample_label(row: pd.Series) -> str:
    option_type = row.get("OptionType", "?")
    maturity = float(row.get("TimeToExpiration", 0.0))
    moneyness = float(row.get("Moneyness", 0.0)) if "Moneyness" in row else float("nan")
    return f"{row.name} | {option_type} | T={maturity:.1f}d | M={moneyness:.3f}"


def display_feature_label(feature_name: str, feature_schema: FeatureSchema) -> str:
    if feature_name in feature_schema.names():
        return feature_schema.get(feature_name).label

    if "__" not in feature_name:
        return feature_name

    _, transformed_name = feature_name.split("__", 1)
    if transformed_name in feature_schema.names():
        return feature_schema.get(transformed_name).label

    for feature in feature_schema.categorical_features(raw_only=True):
        prefix = f"{feature.name}_"
        if transformed_name.startswith(prefix):
            category_value = transformed_name[len(prefix):]
            return f"{feature.label} = {category_value}"

    for feature in feature_schema.numerical_features(raw_only=False):
        if transformed_name == feature.name:
            return feature.label

    return transformed_name


__all__ = [
    "add_derived_features",
    "apply_feature_override",
    "build_sample_label",
    "display_feature_label",
]
