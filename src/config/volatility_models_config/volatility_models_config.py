import json
from pathlib import Path

from src.config.volatility_models_config.training_data_config import TrainingDataConfig


class VolatilityModelsConfig:
    def __init__(self, volatility_models_config_file_path: Path):
        with open(volatility_models_config_file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        self.training_data_config = TrainingDataConfig(volatility_models_config_file_path)
