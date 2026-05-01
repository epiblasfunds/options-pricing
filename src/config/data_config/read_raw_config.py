import json
import typing as t

from src.enums.data_enums import CcontractsC2Enum, DataTypeEnum, TgentradesEnum


class ReadRawConfig:
    def _load_read_raw_step(self, data_config: t.Dict):
        read_raw_step_name = "read_raw_step"

        # CCONTRACTS C2
        cconctracts_c2_columns_dict = data_config[read_raw_step_name][
            "ccontracts_c2_columns"
        ]
        self.ccontracts_c2_columns_list = [
            CcontractsC2Enum(k) for k in cconctracts_c2_columns_dict.keys()
        ]
        self.ccontracts_c2_columns_selected_dict = {
            CcontractsC2Enum(k): DataTypeEnum(v)
            for k, v in cconctracts_c2_columns_dict.items()
            if v is not None
        }
        self.cconctracts_c2_prefix = data_config[read_raw_step_name][
            "ccontracts_c2_prefix"
        ]

        # TGENTRADES
        tgentrades_columns_dict = data_config[read_raw_step_name]["tgentrades_columns"]
        self.tgentrades_columns_list = [
            TgentradesEnum(k) for k in tgentrades_columns_dict.keys()
        ]
        self.tgentrades_columns_selected_dict = {
            TgentradesEnum(k): DataTypeEnum(v)
            for k, v in tgentrades_columns_dict.items()
            if v is not None
        }
        self.tgentrades_prefix = data_config[read_raw_step_name]["tgentrades_prefix"]

        # RATES
        self.idx_rate_values = data_config[read_raw_step_name]["idx_rate_values"]
        self.spread_str_eonia = data_config[read_raw_step_name]["spread_str_eonia"]
        self.cutoff_date_str_eonia = data_config[read_raw_step_name][
            "cutoff_date_str_eonia"
        ]
        self.rates_columns = data_config[read_raw_step_name]["rates_columns"]
        self.rates_date_column_name = data_config[read_raw_step_name][
            "rates_date_column_name"
        ]
        self.rates_output_filename = data_config[read_raw_step_name][
            "rates_output_filename"
        ]

    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        self.first_year = data_config["first_year"]
        self.last_year = data_config["last_year"]

        self._load_read_raw_step(data_config=data_config)

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
