"""Feature derivation helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureSchema,
)


def add_derived_features(frame: pd.DataFrame, feature_schema: FeatureSchema) -> pd.DataFrame:
    """Derive explainability-friendly features without changing raw model inputs."""

    derived = frame.copy()
    if "ExecDatetime" in derived.columns:
        exec_dt = pd.to_datetime(derived["ExecDatetime"], format="mixed", errors="coerce")
        if "ExecHour" in feature_schema.names():
            derived["ExecHour"] = exec_dt.dt.hour.astype("Int64")
        if "ExecWeekday" in feature_schema.names():
            derived["ExecWeekday"] = (exec_dt.dt.weekday + 1).astype("Int64")

    if {"UnderlyingPrice", "StrikePrice"}.issubset(derived.columns):
        ratio = derived["UnderlyingPrice"].astype(float) / derived["StrikePrice"].astype(float)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        if "Moneyness" in feature_schema.names():
            derived["Moneyness"] = ratio
        if "LogMoneyness" in feature_schema.names():
            safe_ratio = ratio.where(ratio > 0.0)
            derived["LogMoneyness"] = np.log(safe_ratio)
        if "AbsLogMoneyness" in feature_schema.names():
            safe_ratio = ratio.where(ratio > 0.0)
            derived["AbsLogMoneyness"] = np.abs(np.log(safe_ratio))

    return derived


def apply_feature_override(frame: pd.DataFrame, feature_name: str, value: Any) -> pd.DataFrame:
    """Override one feature, including supported derived features."""

    updated = frame.copy()
    if feature_name in updated.columns:
        updated[feature_name] = value
        return updated

    if feature_name == "Moneyness":
        updated["StrikePrice"] = updated["UnderlyingPrice"].astype(float) / float(value)
        updated["Moneyness"] = float(value)
        updated["LogMoneyness"] = math.log(float(value))
        updated["AbsLogMoneyness"] = abs(math.log(float(value)))
        return updated

    if feature_name == "LogMoneyness":
        moneyness = math.exp(float(value))
        updated["StrikePrice"] = updated["UnderlyingPrice"].astype(float) / moneyness
        updated["Moneyness"] = moneyness
        updated["LogMoneyness"] = float(value)
        updated["AbsLogMoneyness"] = abs(float(value))
        return updated

    raise KeyError(f"Unsupported override feature: {feature_name}")


def build_sample_label(row: pd.Series) -> str:
    """Human-friendly row label for dropdowns."""

    option_type = row.get("OptionType", "?")
    maturity = float(row.get("TimeToExpiration", 0.0))
    moneyness = float(row.get("Moneyness", 0.0)) if "Moneyness" in row else float("nan")
    return f"{row.name} | {option_type} | T={maturity:.1f}d | M={moneyness:.3f}"


def display_feature_label(feature_name: str, feature_schema: FeatureSchema) -> str:
    """Map raw or transformed feature names to dashboard-friendly labels."""

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

    for feature in feature_schema.numerical_features(raw_only=True):
        if transformed_name == feature.name:
            return feature.label

    return transformed_name
