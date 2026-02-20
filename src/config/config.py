from pathlib import Path

from src.config.data_config.data_config import DataConfig

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

SRC_DIR_PATH = PROJECT_ROOT_PATH / "src"

DATA_DIR_PATH = PROJECT_ROOT_PATH / "data"
SOURCE_DATA_DIR_PATH = DATA_DIR_PATH / "source_data"
RAW_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "raw_data"
MERGE_RAW_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "merge_raw_data"
PRODUCT_SPLIT_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "product_split_data"
UNDERLYING_DATA_STEP_DIR_PATH = DATA_DIR_PATH / "underlying_data"

RESOURCES_PATH = PROJECT_ROOT_PATH / "resources"
DATA_CONFIG_FILE_PATH = RESOURCES_PATH / "data_config.json"


class Config:
    def __init__(self):
        self.data_config = DataConfig(data_config_file_path=DATA_CONFIG_FILE_PATH)


config = Config()
