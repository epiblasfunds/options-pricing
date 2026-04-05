"""Explainability dashboard package for volatility models."""

from src.volatility_models.model_explainability.config import (
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_METRICS_REGISTRY,
    DEFAULT_SETTINGS,
)


def create_app():
    """Lazily import and create the Dash application."""

    from src.volatility_models.model_explainability.app import create_app as _create_app

    return _create_app()


__all__ = ["create_app", "DEFAULT_FEATURE_SCHEMA", "DEFAULT_METRICS_REGISTRY", "DEFAULT_SETTINGS"]
