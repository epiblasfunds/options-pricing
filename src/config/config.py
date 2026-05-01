from pathlib import Path

from src.config.clientserver_config.clientserver_config import ClientserverConfig
from src.config.dashboard_models_config.dashboard_models_config import (
    DashboardModelsConfig,
)
from src.config.data_config.data_config import DataConfig
from src.config.logging_config.logging_config import LoggingConfig
from src.config.volatility_models_config.volatility_models_config import (
    VolatilityModelsConfig,
)

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

SRC_DIR_PATH = PROJECT_ROOT_PATH / "src"
VOLATILITY_MODELS_DIR_PATH = SRC_DIR_PATH / "volatility_models"
DASHBOARD_DIR_PATH = SRC_DIR_PATH / "dashboard"
DASHBOARD_SAVED_MODELS_DIR_PATH = DASHBOARD_DIR_PATH / "saved_models"
VOLATILITY_TRAINED_MODELS_DIR_PATH = VOLATILITY_MODELS_DIR_PATH / "trained_models"
VOLATILITY_METADATA_DIR_PATH = VOLATILITY_MODELS_DIR_PATH / "trained_metadata"
VOLATILITY_FAMILY_METADATA_DIR_PATH = VOLATILITY_METADATA_DIR_PATH / "family_metadata"
VOLATILITY_RETRAINED_METADATA_DIR_PATH = (
    VOLATILITY_METADATA_DIR_PATH / "retrained_metadata"
)

DATA_DIR_PATH = PROJECT_ROOT_PATH / "data"
SOURCE_DATA_DIR_PATH = DATA_DIR_PATH / "source_data"
SOURCE_MARKET_DATA_DIR_PATH = SOURCE_DATA_DIR_PATH / "market_data"
SOURCE_RATES_DATA_DIR_PATH = SOURCE_DATA_DIR_PATH / "rates_data"
RAW_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "raw_data"
MERGE_RAW_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "merge_raw_data"
PRODUCT_SPLIT_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "product_split_data"
UNDERLYING_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "underlying_data"
VOLATILITY_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "volatility_data"

TRAINING_DATA_DIR = DATA_DIR_PATH / "training_data"
TRAINING_DATA_SPLITTED_DIR = TRAINING_DATA_DIR / "splitted"
TRAINING_DATA_SPLITTED_FEATURES_DIR = TRAINING_DATA_DIR / "splitted_features"
TRAINING_DATA_KFOLDS_DIR = TRAINING_DATA_DIR / "kfolds"

RESOURCES_PATH = PROJECT_ROOT_PATH / "resources"

for pth in [
    SRC_DIR_PATH,
    DATA_DIR_PATH,
    SOURCE_DATA_DIR_PATH,
    RAW_DATA_STEP_DIR_PATH,
    MERGE_RAW_DATA_STEP_DIR_PATH,
    PRODUCT_SPLIT_DATA_STEP_DIR_PATH,
    UNDERLYING_DATA_STEP_DIR_PATH,
    VOLATILITY_DATA_STEP_DIR_PATH,
    DASHBOARD_SAVED_MODELS_DIR_PATH,
    VOLATILITY_TRAINED_MODELS_DIR_PATH,
    VOLATILITY_FAMILY_METADATA_DIR_PATH,
    VOLATILITY_RETRAINED_METADATA_DIR_PATH,
    TRAINING_DATA_DIR,
    TRAINING_DATA_SPLITTED_DIR,
    TRAINING_DATA_SPLITTED_FEATURES_DIR,
    TRAINING_DATA_KFOLDS_DIR,
    RESOURCES_PATH,
]:
    pth.mkdir(parents=True, exist_ok=True)


class Config:
    CLIENTSERVER_CONFIG_FILE_PATH = RESOURCES_PATH / "clientserver.json"
    DATA_CONFIG_FILE_PATH = RESOURCES_PATH / "data_config.json"
    DASHBOARD_MODELS_CONFIG_FILE_PATH = RESOURCES_PATH / "dashboard_models_config.json"
    VOLATILITY_MODELS_CONFIG_FILE_PATH = (
        RESOURCES_PATH / "volatility_models_config.json"
    )

    def __init__(self):
        self.clientserver_config = ClientserverConfig(
            clientserver_config_file_path=Config.CLIENTSERVER_CONFIG_FILE_PATH
        )
        self.data_config = DataConfig(
            data_config_file_path=Config.DATA_CONFIG_FILE_PATH
        )
        self.dashboard_models_config = DashboardModelsConfig(
            dashboard_models_config_file_path=Config.DASHBOARD_MODELS_CONFIG_FILE_PATH
        )
        self.volatility_models_config = VolatilityModelsConfig(
            volatility_models_config_file_path=Config.VOLATILITY_MODELS_CONFIG_FILE_PATH
        )
        LoggingConfig.setup_logging()


config = Config()
