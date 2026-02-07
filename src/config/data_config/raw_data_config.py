import json
import typing as t

from src.enums.data_type import DataType


class RawDataConfig:
    def _load_read_raw_step(self, data_config: t.Dict):
        read_raw_step_name = "read_raw_step"

        # CCONTRACTS C2
        cconctracts_c2_columns_dict = data_config[read_raw_step_name][
            "ccontracts_c2_columns"
        ]
        self.ccontracts_c2_columns_list = list(cconctracts_c2_columns_dict.keys())
        self.ccontracts_c2_columns_selected_dict = {
            k: DataType[v]
            for k, v in cconctracts_c2_columns_dict.items()
            if v is not None
        }
        self.cconctracts_c2_prefix = data_config[read_raw_step_name][
            "ccontracts_c2_prefix"
        ]

        # TGENTRADES
        tgentrades_columns_dict = data_config[read_raw_step_name]["tgentrades_columns"]
        self.tgentrades_columns_list = list(tgentrades_columns_dict.keys())
        self.tgentrades_columns_selected_dict = {
            k: DataType[v] for k, v in tgentrades_columns_dict.items() if v is not None
        }
        self.tgentrades_prefix = data_config[read_raw_step_name]["tgentrades_prefix"]

    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        self.first_year = data_config["first_year"]
        self.last_year = data_config["last_year"]

        self._load_read_raw_step(data_config=data_config)

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
