import json


class UnderlyingConfig:
    def _load_config(self, data_config_file_path: str):
        with(open(data_config_file_path, "r") as f):
            data_config = json.load(f)
        underlying_step_name = "underlying_step"
        self.options_trade_underlying_ibex_database_columns = data_config[underlying_step_name]["options_trade_underlying_ibex_database_columns"]
        self.output_filename = data_config[underlying_step_name]["output_filename"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
