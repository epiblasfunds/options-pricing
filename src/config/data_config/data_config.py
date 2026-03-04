from src.config.data_config.contract_code_config import ContractCodeConfig
from src.config.data_config.merge_raw_config import MergeRawConfig
from src.config.data_config.product_split_config import ProductSplitConfig
from src.config.data_config.read_rates_raw_config import ReadRatesRawConfig
from src.config.data_config.read_raw_config import ReadRawConfig
from src.config.data_config.underlying_config import UnderlyingConfig
from src.config.data_config.underlying_rates_config import UnderlyingRatesConfig
from src.config.data_config.volatility_config import VolatilityConfig


class DataConfig:
    def __init__(self, data_config_file_path: str):
        self.contract_code_config = ContractCodeConfig(
            data_config_file_path=data_config_file_path
        )
        self.read_raw_config = ReadRawConfig(
            data_config_file_path=data_config_file_path
        )
        self.merge_raw_config = MergeRawConfig(
            data_config_file_path=data_config_file_path
        )
        self.product_split_config = ProductSplitConfig(
            data_config_file_path=data_config_file_path
        )
        self.underlying_config = UnderlyingConfig(
            data_config_file_path=data_config_file_path
        )
        self.read_rates_raw_config = ReadRatesRawConfig(
            data_config_file_path=data_config_file_path
        )
        self.underlying_rates_config = UnderlyingRatesConfig(
            data_config_file_path=data_config_file_path
        )
        self.volatility_config = VolatilityConfig(
            data_config_file_path=data_config_file_path
        )
