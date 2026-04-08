from pathlib import Path

import pandas as pd

from src.config.config import VOLATILITY_TRAINED_MODELS_DIR_PATH
from src.data_management.loaders.volatility_step_loader import VolatilityStepLoader
from src.volatility_models import TrainedModelMetadata
from src.volatility_models import select_trade_columns


def load_reference_trade_frame(force_reload: bool = False) -> pd.DataFrame:
    frame = VolatilityStepLoader.load(force_reload=force_reload)
    selected = select_trade_columns(frame)
    return selected.sort_values("ExecDatetime").reset_index(drop=True)


def discover_trained_model_metadata(
    trained_models_dir: Path = VOLATILITY_TRAINED_MODELS_DIR_PATH,
) -> list[TrainedModelMetadata]:
    metadata_items: list[TrainedModelMetadata] = []
    for artifact in sorted(trained_models_dir.iterdir(), key=lambda path: path.name):
        metadata_path = artifact / "metadata.json"
        if artifact.is_dir() and metadata_path.exists():
            metadata_items.append(TrainedModelMetadata.load(artifact))
    return metadata_items
