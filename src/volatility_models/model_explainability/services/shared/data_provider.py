"""Access to the repository volatility dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureSchema,
)
from src.volatility_models.model_explainability.utils.feature_utils import (
    add_derived_features,
)
from src.volatility_models.model_explainability.utils.validation import (
    ensure_non_empty_frame,
)

logger = logging.getLogger(__name__)


class VolatilityDataProvider:
    """Read and cache the explainability dataset."""

    def __init__(self, dataset_path: Path, feature_schema: FeatureSchema) -> None:
        self.dataset_path = dataset_path
        self.feature_schema = feature_schema
        self._cache: pd.DataFrame | None = None

    def load_dataset(self, refresh: bool = False) -> pd.DataFrame:
        if self._cache is not None and not refresh:
            return self._cache.copy()

        if self.dataset_path.exists():
            frame = pd.read_csv(self.dataset_path, sep=";", low_memory=False)
        else:
            logger.info("Volatility dataset not found at %s. Running loader.", self.dataset_path)
            from src.data_management.loaders.volatility_step_loader import (
                VolatilityStepLoader,
            )

            frame = VolatilityStepLoader.load(force_reload=False)

        frame = add_derived_features(frame, self.feature_schema)
        ensure_non_empty_frame(frame, "The volatility dataset is empty.")
        self._cache = frame.copy()
        return frame
