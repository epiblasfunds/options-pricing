import json
import typing as t
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
RESOURCES_PATH = PROJECT_ROOT_PATH / "resources"

DATA_CONFIG_FILE_PATH = RESOURCES_PATH / "data_config.json"


class Config:

    def _load_read_raw_step(self, data_config: t.Dict):
        read_raw_step_name = "read_raw_step"

        # CCONTRACTS C2
        cconctracts_c2_columns_dict = data_config[read_raw_step_name][
            "cconctracts_c2_columns"
        ]
        self.ccontracts_c2_columns_list = list(cconctracts_c2_columns_dict.keys())
        self.ccontracts_c2_columns_selected_dict = {
            k: v for k, v in cconctracts_c2_columns_dict.items() if v is not None
        }
        self.cconctracts_c2_prefix = data_config[read_raw_step_name][
            "cconctracts_c2_prefix"
        ]

        # TGENTRADES
        tgentrades_columns_dict = data_config[read_raw_step_name]["tgentrades_columns"]
        self.tgentrades_columns_list = list(tgentrades_columns_dict.keys())
        self.tgentrades_columns_selected_dict = {
            k: v for k, v in tgentrades_columns_dict.items() if v is not None
        }
        self.tgentrades_prefix = data_config[read_raw_step_name]["tgentrades_prefix"]

    def _load_data(self):
        with open(DATA_CONFIG_FILE_PATH, "r") as f:
            data_config = json.load(f)

        self.data_first_year = data_config["first_year"]
        self.data_last_year = data_config["last_year"]

        self._load_read_raw_step(data_config=data_config)

    def __init__(self):
        self._load_config()


config = Config()
