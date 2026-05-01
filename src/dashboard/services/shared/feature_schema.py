"""Feature schema definitions and helpers."""

from dataclasses import dataclass
from statistics import multimode
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata about a single model or explainability feature."""

    name: str
    label: str
    dtype: str
    category: str
    raw_input: bool = True
    derived_explainability_feature: bool = False
    allowed_values: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    default_value: Any | None = None
    widget: str | None = None
    description: str | None = None

    @property
    def is_categorical(self) -> bool:
        return self.category == "categorical"

    @property
    def is_numerical(self) -> bool:
        return self.category == "numerical"


class FeatureSchema:
    """Container for feature metadata."""

    def __init__(self, features: list[FeatureDefinition], target_column: str):
        self._features = tuple(features)
        self._by_name = {feature.name: feature for feature in features}
        self.target_column = target_column

    @property
    def features(self) -> tuple[FeatureDefinition, ...]:
        return self._features

    def get(self, name: str) -> FeatureDefinition:
        return self._by_name[name]

    def names(self) -> list[str]:
        return [feature.name for feature in self._features]

    def raw_input_features(self) -> list[FeatureDefinition]:
        return [feature for feature in self._features if feature.raw_input]

    def explainability_features(self) -> list[FeatureDefinition]:
        return [
            feature
            for feature in self._features
            if feature.raw_input or feature.derived_explainability_feature
        ]

    def numerical_features(self, raw_only: bool = False) -> list[FeatureDefinition]:
        features = (
            self.raw_input_features() if raw_only else self.explainability_features()
        )
        return [feature for feature in features if feature.is_numerical]

    def categorical_features(self, raw_only: bool = False) -> list[FeatureDefinition]:
        features = (
            self.raw_input_features() if raw_only else self.explainability_features()
        )
        return [feature for feature in features if feature.is_categorical]

    def labels(self, names: list[str] | None = None) -> dict[str, str]:
        selected_names = names or self.names()
        return {name: self.get(name).label for name in selected_names}

    def defaults_from_frame(
        self,
        frame: pd.DataFrame,
        raw_only: bool = True,
    ) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        features = (
            self.raw_input_features() if raw_only else self.explainability_features()
        )
        for feature in features:
            if feature.default_value is not None:
                defaults[feature.name] = feature.default_value
                continue
            if feature.name not in frame.columns or frame.empty:
                defaults[feature.name] = None
                continue
            series = frame[feature.name].dropna()
            if series.empty:
                defaults[feature.name] = None
            elif feature.is_numerical:
                defaults[feature.name] = float(series.median())
            else:
                defaults[feature.name] = multimode(series.tolist())[0]
        return defaults

    def normalize_value(self, feature_name: str, value: Any) -> Any:
        feature = self.get(feature_name)
        normalized = value.value if hasattr(value, "value") else value

        if feature.dtype == "datetime":
            if normalized in (None, ""):
                return normalized
            return pd.to_datetime(normalized, format="mixed", errors="coerce")

        if feature.is_numerical:
            if normalized in (None, ""):
                return normalized
            try:
                numeric_value = float(normalized)
            except (TypeError, ValueError):
                return normalized
            if feature.dtype == "int":
                return int(round(numeric_value))
            return numeric_value

        if normalized in (None, ""):
            return normalized

        if feature.allowed_values is None:
            return normalized

        if normalized in feature.allowed_values:
            return normalized

        text_value = str(normalized).strip()
        allowed_by_text = {str(item): item for item in feature.allowed_values}
        if text_value in allowed_by_text:
            return allowed_by_text[text_value]

        uppercase_text = text_value.upper()
        uppercase_allowed = {str(item).upper(): item for item in feature.allowed_values}
        if uppercase_text in uppercase_allowed:
            return uppercase_allowed[uppercase_text]

        enum_suffix = uppercase_text.split(".")[-1]
        option_aliases = {"CALL": "C", "PUT": "P"}
        if (
            enum_suffix in option_aliases
            and option_aliases[enum_suffix] in uppercase_allowed
        ):
            return uppercase_allowed[option_aliases[enum_suffix]]

        if all(isinstance(item, int) for item in feature.allowed_values):
            try:
                numeric_key = str(int(float(text_value)))
            except (TypeError, ValueError):
                return normalized
            return allowed_by_text.get(numeric_key, normalized)

        return normalized

    def normalize_sample(
        self,
        sample: dict[str, Any],
        required_raw_inputs: bool = True,
    ) -> dict[str, Any]:
        features = (
            self.raw_input_features()
            if required_raw_inputs
            else self.explainability_features()
        )
        normalized: dict[str, Any] = dict(sample)
        for feature in features:
            if feature.name in normalized:
                normalized[feature.name] = self.normalize_value(
                    feature.name, normalized[feature.name]
                )
        return normalized

    def validate_sample(
        self,
        sample: dict[str, Any],
        required_raw_inputs: bool = True,
    ) -> dict[str, str]:
        errors: dict[str, str] = {}
        features = (
            self.raw_input_features()
            if required_raw_inputs
            else self.explainability_features()
        )
        normalized_sample = self.normalize_sample(
            sample, required_raw_inputs=required_raw_inputs
        )
        for feature in features:
            if feature.name not in normalized_sample or normalized_sample[
                feature.name
            ] in (None, ""):
                errors[feature.name] = "Value is required."
                continue
            value = normalized_sample[feature.name]
            if (
                feature.allowed_values is not None
                and value not in feature.allowed_values
            ):
                errors[feature.name] = "Value is outside the allowed set."
                continue
            if feature.is_numerical:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    errors[feature.name] = "Value must be numeric."
                    continue
                if feature.min_value is not None and numeric_value < feature.min_value:
                    errors[feature.name] = "Value is below the minimum."
                if feature.max_value is not None and numeric_value > feature.max_value:
                    errors[feature.name] = "Value is above the maximum."
            elif feature.dtype == "datetime" and pd.isna(value):
                errors[feature.name] = "Value must be a valid datetime."
        return errors
