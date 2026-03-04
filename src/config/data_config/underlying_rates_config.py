import json


class UnderlyingRatesConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        underlying_rates_step_name = "underlying_rates_step"
        rates_config = data_config[underlying_rates_step_name]

        self.tenors_days = rates_config["tenors_days"]
        self.output_filename = rates_config["output_filename"]
        
    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
