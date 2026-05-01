"""Feature helpers shared by the dashboard and bundle exporter."""

import pandas as pd

from src.dashboard.services.shared.feature_schema import FeatureSchema


def build_sample_label(row: pd.Series) -> str:
    option_type = _display_option_type(row.get("OptionType", "?"))
    strike = _display_number(row.get("StrikePrice"))
    maturity = float(row.get("TimeToExpiration", 0.0))
    moneyness = float(row.get("Moneyness", 0.0)) if "Moneyness" in row else float("nan")
    return (
        f"Sample ID: {row.name} | Option type: {option_type} | "
        f"Strike: {strike} | Time to expiration: {maturity:.1f} days | "
        f"Moneyness: {moneyness:.3f}"
    )


def build_manual_input_sample_label(row: pd.Series) -> str:
    exec_datetime = _display_datetime(row.get("ExecDatetime"))
    option_type = _display_option_type(row.get("OptionType", "?"))
    strike = _display_number(row.get("StrikePrice"))
    underlying = _display_number(row.get("UnderlyingPrice"))
    maturity = _display_number(row.get("TimeToExpiration"), decimals=1)
    rate = _display_number(row.get("Rate"), decimals=2)
    return (
        f"ID: {row.name} | Exec datetime: {exec_datetime} | "
        f"Type: {option_type} | Strike: {strike} | "
        f"Underlying: {underlying} | Time to expiration: {maturity} | "
        f"Rate: {rate}"
    )


def _display_option_type(value) -> str:
    text = str(value.value if hasattr(value, "value") else value).strip().upper()
    if text == "C":
        return "CALL"
    if text == "P":
        return "PUT"
    return text or "?"


def _display_datetime(value) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "?"
    return timestamp.isoformat(sep=" ", timespec="seconds")


def _display_number(value, decimals: int = 0) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "?"
    return f"{numeric:,.{decimals}f}"


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
            category_value = transformed_name[len(prefix) :]
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
    "build_manual_input_sample_label",
    "build_sample_label",
    "display_feature_label",
    "replace_feature_names_in_text",
]
