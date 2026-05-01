import json


class MergeRawConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)
        merge_raw_step_name = "merge_raw_step"
        self.merge_columns_list = data_config[merge_raw_step_name]["merge_columns"]
        self.trade_ibex_db_columns = data_config[merge_raw_step_name][
            "trade_ibex_db_columns"
        ]
        self.contract_type_column = data_config[merge_raw_step_name][
            "contract_type_column"
        ]
        self.maturity_hour_expiration = data_config[merge_raw_step_name][
            "maturity_hour_expiration"
        ]
        self.output_filename = data_config[merge_raw_step_name]["output_filename"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
