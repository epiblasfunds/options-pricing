"""Access to persisted dashboard data."""

import logging
from pathlib import Path

import pandas as pd

from src.dashboard.services.shared.model_loader import ModelLoader
from src.dashboard.services.shared.model_registry import ModelRegistry
from src.dashboard.utils.validation import ensure_non_empty_frame
from src.data_management.loaders.volatility_step_loader import VolatilityStepLoader
from src.volatility_models import build_model_dataset, select_trade_columns

logger = logging.getLogger(__name__)


class VolatilityDataProvider:
    """Read and cache the dashboard reference dataset."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
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
        if (
            model_id
            and self.model_registry is not None
            and self.model_loader is not None
        ):
            discovered = self.model_registry.get_model(model_id)
            if discovered is not None:
                bundle = self.model_loader.load(discovered)
                if bundle.dashboard_model is not None:
                    return bundle.dashboard_model.dataset_frame.copy()

        if self._cache is not None and not refresh:
            return self._cache.copy()

        if self.dataset_path.exists():
            frame = pd.read_csv(self.dataset_path, sep=";", low_memory=False)
        else:
            logger.info(
                "Persisted dashboard split not found at %s. Falling back to VolatilityStepLoader.",
                self.dataset_path,
            )
            frame = select_trade_columns(VolatilityStepLoader.load(force_reload=False))

        dataset = build_model_dataset(frame)
        ensure_non_empty_frame(dataset, "The volatility dataset is empty.")
        self._cache = dataset.copy()
        return dataset
