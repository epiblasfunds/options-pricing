"""Reusable preprocessing builders."""

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.dashboard.services.shared.feature_schema import FeatureSchema


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibility wrapper for sklearn versions with different keyword names."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_ordinal_encoder() -> OrdinalEncoder:
    """Compatibility wrapper for sklearn versions with different keyword names."""

    try:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
    except TypeError:
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def build_similarity_preprocessor(
    schema: FeatureSchema,
    feature_names: Sequence[str],
) -> ColumnTransformer:
    """Preprocessor for nearest-neighbor distance computations."""

    numeric_columns = [
        feature.name
        for feature in schema.numerical_features(raw_only=True)
        if feature.name in feature_names
    ]
    categorical_columns = [
        feature.name
        for feature in schema.categorical_features(raw_only=True)
        if feature.name in feature_names
    ]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(
    schema: FeatureSchema,
    feature_names: Sequence[str],
) -> ColumnTransformer:
    """Preprocessor for tree-based surrogate models."""

    numeric_columns = [
        feature.name
        for feature in schema.numerical_features(raw_only=True)
        if feature.name in feature_names
    ]
    categorical_columns = [
        feature.name
        for feature in schema.categorical_features(raw_only=True)
        if feature.name in feature_names
    ]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_ordinal_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
