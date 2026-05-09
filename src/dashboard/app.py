"""Dash app factory."""

from dataclasses import dataclass
import logging

from dash import Dash

from src.config.config import config
from src.dashboard.dashboard.callbacks import register_callbacks
from src.dashboard.dashboard.layout import build_layout
from src.dashboard.domain import build_feature_schema, build_metrics_registry
from src.dashboard.services.behaviour_surface import SurfaceService
from src.dashboard.services.diagnosis import DiagnosisService
from src.dashboard.services.equivalent_models import EquivalentModelsService
from src.dashboard.services.global_explainability import ShapService
from src.dashboard.services.sample_explainability import NeighborsService
from src.dashboard.services.shared.cache_service import CacheService
from src.dashboard.services.shared.data_provider import VolatilityDataProvider
from src.dashboard.services.shared.model_loader import ModelLoader
from src.dashboard.services.shared.model_registry import ModelRegistry
from src.dashboard.services.shared.prediction_service import PredictionService
from src.dashboard.services.shared.storage_runtime import DashboardModelStorageRuntime

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Runtime service container."""

    storage_runtime: DashboardModelStorageRuntime
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

    feature_schema = build_feature_schema()
    metrics_registry = build_metrics_registry()
    cache = CacheService(max_entries=config.dashboard_models_config.cache_entries)
    storage_runtime = DashboardModelStorageRuntime(config.clientserver_config)
    model_dir = storage_runtime.prepare_model_dir()
    model_registry = ModelRegistry(model_dir=model_dir)
    logger.info("Dashboard model directory resolved to %s", model_dir)
    if not model_registry.discover_models():
        logger.info(
            "No valid explainable-model bundles were discovered under %s. "
            "Check that each bundle contains metadata.json and dashboard_model/metadata.json.",
            model_dir,
        )
    model_loader = ModelLoader(cache_size=config.dashboard_models_config.cache_entries)
    data_provider = VolatilityDataProvider()
    data_provider.bind_model_runtime(
        model_registry=model_registry,
        model_loader=model_loader,
    )
    prediction_service = PredictionService(
        model_registry=model_registry,
        model_loader=model_loader,
        feature_schema=feature_schema,
    )
    surface_service = SurfaceService(
        prediction_service=prediction_service,
        feature_schema=feature_schema,
    )
    return Services(
        storage_runtime=storage_runtime,
        feature_schema=feature_schema,
        metrics_registry=metrics_registry,
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
        ),
        neighbors_service=NeighborsService(
            prediction_service=prediction_service,
            feature_schema=feature_schema,
        ),
        surface_service=surface_service,
        diagnosis_service=DiagnosisService(
            prediction_service=prediction_service,
        ),
    )


def create_app():
    """Create the Dash application."""

    services = build_services()
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="Volatility Explainability",
        external_scripts=[
            "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        ],
    )
    app.layout = build_layout()
    register_callbacks(app, services)
    return app
