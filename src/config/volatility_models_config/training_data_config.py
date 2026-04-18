import json
import typing as t

from src.config.volatility_models_config.kfolds_config import KFoldsConfig
from src.config.volatility_models_config.train_test_split_config import (
    TrainTestSplitConfig,
)
from src.enums.data_enums import TrainingDataEnum, VolatilityDBEnum


class TrainingDataConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        training_data_config = data_config["training_data_config"]

        self.vol_db_cols = [
            VolatilityDBEnum(c).value for c in training_data_config["vol_db_cols"]
        ]
        self.aux_context_cols = [
            VolatilityDBEnum(c).value for c in training_data_config["aux_context_cols"]
        ]
        self.raw_model_input = [
            VolatilityDBEnum(c).value for c in training_data_config["raw_model_input"]
        ]
        self.numeric_features = [
            TrainingDataEnum(c).value for c in training_data_config["numeric_features"]
        ]
        self.target_column = TrainingDataEnum(training_data_config["target_column"]).value
        self.trade_type_to_feature = {
            k: TrainingDataEnum(v).value
            for k, v in training_data_config["trade_type_to_feature"].items()
        }
        self.train_test_split_config = TrainTestSplitConfig(data_config_file_path)
        self.kfolds_config = KFoldsConfig(data_config_file_path)
        self.custom_error_1: t.Dict = training_data_config["custom_error_1"]
        self.custom_error_2: t.Dict = training_data_config["custom_error_2"]
        self.models_metrics: t.List = training_data_config["models_metrics"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
