"""Feature helpers shared by the dashboard and bundle exporter."""

import pandas as pd

from src.dashboard.services.shared.feature_schema import FeatureSchema


def build_sample_label(row: pd.Series) -> str:
    option_type = _display_option_type(row.get("OptionType", "?"))
    maturity = float(row.get("TimeToExpiration", 0.0))
    moneyness = float(row.get("Moneyness", 0.0)) if "Moneyness" in row else float("nan")
    return (
        f"Sample ID: {row.name} | Option type: {option_type} | "
        f"Time to expiration: {maturity:.1f} days | Moneyness: {moneyness:.3f}"
    )


def _display_option_type(value) -> str:
    text = str(value.value if hasattr(value, "value") else value).strip().upper()
    if text == "C":
        return "CALL"
    if text == "P":
        return "PUT"
    return text or "?"


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


def replace_feature_names_in_text(text: str, feature_schema: FeatureSchema) -> str:
    updated = str(text)
    for feature_name in sorted(feature_schema.names(), key=len, reverse=True):
        updated = updated.replace(
            feature_name,
            display_feature_label(feature_name, feature_schema),
        )
    return updated


__all__ = [
    "build_sample_label",
    "display_feature_label",
    "replace_feature_names_in_text",
]
