"""Access to persisted dashboard data."""

import logging

import pandas as pd

from src.dashboard.services.shared.model_loader import ModelLoader
from src.dashboard.services.shared.model_registry import ModelRegistry
from src.dashboard.utils.validation import ensure_non_empty_frame

logger = logging.getLogger(__name__)


class VolatilityDataProvider:
    """Read and cache the dashboard reference dataset."""

    def __init__(self) -> None:
        self._cache: pd.DataFrame | None = None
        self.model_registry: ModelRegistry | None = None
        self.model_loader: ModelLoader | None = None

    def bind_model_runtime(
        self,
        model_registry: ModelRegistry,
        model_loader: ModelLoader,
    ) -> None:
        self.model_registry = model_registry
        self.model_loader = model_loader

    def load_dataset(
        self,
        refresh: bool = False,
        model_id: str | None = None,
    ) -> pd.DataFrame:
        if self._cache is not None and not refresh and model_id is None:
            return self._cache.copy()

        if (
            model_id
            and self.model_registry is not None
            and self.model_loader is not None
        ):
            discovered = self.model_registry.get_model(model_id)
            if discovered is not None:
                bundle = self.model_loader.load(discovered)
                if bundle.dashboard_model is not None:
                    dataset = bundle.dashboard_model.dataset_frame.copy()
                    ensure_non_empty_frame(dataset, "The volatility dataset is empty.")
                    return dataset
            raise FileNotFoundError(
                f"Dashboard model bundle '{model_id}' was not found or has no dataset."
            )

        if (
            model_id is None
            and self.model_registry is not None
            and self.model_loader is not None
        ):
            discovered_models = self.model_registry.discover_models()
            if discovered_models:
                first_bundle = self.model_loader.load(discovered_models[0])
                if first_bundle.dashboard_model is not None:
                    dataset = first_bundle.dashboard_model.dataset_frame.copy()
                    ensure_non_empty_frame(dataset, "The volatility dataset is empty.")
                    self._cache = dataset.copy()
                    return dataset

        logger.error("No dashboard model bundles are available for dataset loading.")
        raise FileNotFoundError(
            "No dashboard model bundles are available for dataset loading."
        )

    def clear(self) -> None:
        self._cache = None
