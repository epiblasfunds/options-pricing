from pathlib import Path
from src.config.volatility_models_config.training_data_config import TrainingDataConfig


class VolatilityModelsConfig:
    def __init__(self, volatility_models_config_file_path: Path):
        self.training_data_config = TrainingDataConfig(
            volatility_models_config_file_path
        )
