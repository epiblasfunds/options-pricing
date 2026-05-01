import json


class VolatilityConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        volatility_step_name = "volatility_step"
        volatility_config = data_config[volatility_step_name]

        self.solver_min_sigma = volatility_config["solver_min_sigma"]
        self.solver_max_sigma = volatility_config["solver_max_sigma"]
        self.solver_tol = volatility_config["solver_tol"]
        self.trade_type_filter = volatility_config["trade_type_filter"]
        self.underlying_lag_max_minutes = volatility_config[
            "underlying_lag_max_minutes"
        ]
        self.output_filename = volatility_config["output_filename"]
        self.volatility_db_columns = volatility_config["volatility_db_columns"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
