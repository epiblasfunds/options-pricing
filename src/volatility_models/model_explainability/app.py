"""Dash app factory."""

from __future__ import annotations

from dataclasses import dataclass

from src.volatility_models.model_explainability.config import (
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_METRICS_REGISTRY,
    DEFAULT_SETTINGS,
)
from src.volatility_models.model_explainability.dashboard.callbacks import register_callbacks
from src.volatility_models.model_explainability.dashboard.layout import build_layout
from src.volatility_models.model_explainability.services.behaviour_surface import (
    SurfaceService,
)
from src.volatility_models.model_explainability.services.diagnosis import (
    DiagnosisService,
)
from src.volatility_models.model_explainability.services.equivalent_models import (
    EquivalentModelsService,
)
from src.volatility_models.model_explainability.services.global_explainability import (
    ShapService,
)
from src.volatility_models.model_explainability.services.sample_explainability import (
    NeighborsService,
)
from src.volatility_models.model_explainability.services.shared.cache_service import (
    CacheService,
)
from src.volatility_models.model_explainability.services.shared.data_provider import (
    VolatilityDataProvider,
)
from src.volatility_models.model_explainability.services.shared.model_loader import (
    ModelLoader,
)
from src.volatility_models.model_explainability.services.shared.model_registry import (
    ModelRegistry,
)
from src.volatility_models.model_explainability.services.shared.prediction_service import (
    PredictionService,
)


@dataclass
class Services:
    """Runtime service container."""

    settings: object
    feature_schema: object
    metrics_registry: object
    cache: CacheService
    model_registry: ModelRegistry
    model_loader: ModelLoader
    data_provider: VolatilityDataProvider
    prediction_service: PredictionService
    equivalent_models_service: EquivalentModelsService
    shap_service: ShapService
    neighbors_service: NeighborsService
    surface_service: SurfaceService
    diagnosis_service: DiagnosisService


def build_services() -> Services:
    """Create the service container."""

    cache = CacheService(max_entries=DEFAULT_SETTINGS.cache_entries)
    model_registry = ModelRegistry(DEFAULT_SETTINGS.model_dir)
    model_loader = ModelLoader()
    data_provider = VolatilityDataProvider(
        dataset_path=DEFAULT_SETTINGS.volatility_dataset_path,
        feature_schema=DEFAULT_FEATURE_SCHEMA,
    )
    prediction_service = PredictionService(
        model_registry=model_registry,
        model_loader=model_loader,
        feature_schema=DEFAULT_FEATURE_SCHEMA,
    )
    surface_service = SurfaceService(
        prediction_service=prediction_service,
        feature_schema=DEFAULT_FEATURE_SCHEMA,
    )
    return Services(
        settings=DEFAULT_SETTINGS,
        feature_schema=DEFAULT_FEATURE_SCHEMA,
        metrics_registry=DEFAULT_METRICS_REGISTRY,
        cache=cache,
        model_registry=model_registry,
        model_loader=model_loader,
        data_provider=data_provider,
        prediction_service=prediction_service,
        equivalent_models_service=EquivalentModelsService(
            prediction_service=prediction_service
        ),
        shap_service=ShapService(
            prediction_service=prediction_service,
            feature_schema=DEFAULT_FEATURE_SCHEMA,
        ),
        neighbors_service=NeighborsService(
            prediction_service=prediction_service,
            feature_schema=DEFAULT_FEATURE_SCHEMA,
        ),
        surface_service=surface_service,
        diagnosis_service=DiagnosisService(
            prediction_service=prediction_service,
            metrics_registry=DEFAULT_METRICS_REGISTRY,
            surface_service=surface_service,
            target_column=DEFAULT_SETTINGS.target_column,
        ),
    )


def create_app():
    """Create the Dash application."""

    try:
        from dash import Dash
    except ImportError as exc:
        raise ImportError(
            "Dash is required to run the explainability dashboard."
        ) from exc

    services = build_services()
    app = Dash(__name__, suppress_callback_exceptions=True, title="Volatility Explainability")
    app.layout = build_layout()
    register_callbacks(app, services)
    return app
