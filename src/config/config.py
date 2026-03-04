from pathlib import Path

from src.config.data_config.data_config import DataConfig
from src.config.logging_config.logging_config import LoggingConfig

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

SRC_DIR_PATH = PROJECT_ROOT_PATH / "src"

DATA_DIR_PATH = PROJECT_ROOT_PATH / "data"
SOURCE_DATA_DIR_PATH = DATA_DIR_PATH / "source_data"
RAW_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "raw_data"
MERGE_RAW_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "merge_raw_data"
PRODUCT_SPLIT_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "product_split_data"
UNDERLYING_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "underlying_data"
RISK_FREE_RATES_DATA_DIR_PATH = DATA_DIR_PATH / "risk_free_rates_data"
READ_RATES_RAW_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "read_rates_raw_data"
UNDERLYING_RATES_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "underlying_rates_data"
VOLATILITY_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "volatility_data"

RESOURCES_PATH = PROJECT_ROOT_PATH / "resources"

for pth in [
    SRC_DIR_PATH,
    DATA_DIR_PATH,
    SOURCE_DATA_DIR_PATH,
    RAW_DATA_STEP_DIR_PATH,
    MERGE_RAW_DATA_STEP_DIR_PATH,
    PRODUCT_SPLIT_DATA_STEP_DIR_PATH,
    UNDERLYING_DATA_STEP_DIR_PATH,
    READ_RATES_RAW_DATA_STEP_DIR_PATH,
    UNDERLYING_RATES_DATA_STEP_DIR_PATH,
    VOLATILITY_DATA_STEP_DIR_PATH,
    RESOURCES_PATH,
]:
    pth.mkdir(parents=True, exist_ok=True)


class Config:
    DATA_CONFIG_FILE_PATH = RESOURCES_PATH / "data_config.json"

    def __init__(self):
        self.data_config = DataConfig(data_config_file_path=Config.DATA_CONFIG_FILE_PATH)
        LoggingConfig.setup_logging()


config = Config()
