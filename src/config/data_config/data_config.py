from src.config.data_config.raw_data_config import RawDataConfig


class DataConfig:
    def __init__(self, data_config_file_path: str):
        self.raw_data_config = RawDataConfig(
            data_config_file_path=data_config_file_path
        )
